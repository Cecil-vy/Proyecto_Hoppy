import sys
import mujoco
import mujoco.viewer
import numpy as np

# Modelo
model = mujoco.MjModel.from_xml_path("hoppy.xml")
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# IDs de articulaciones
hip_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
knee_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee")
HIP_DOF = model.jnt_dofadr[hip_jnt_id]
KNEE_DOF = model.jnt_dofadr[knee_jnt_id]
HIP_QADR = model.jnt_qposadr[hip_jnt_id]
KNEE_QADR = model.jnt_qposadr[knee_jnt_id]

yaw_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yaw")
YAW_DOF = model.jnt_dofadr[yaw_jnt_id]

BOOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "boom_hip")
FOOT_SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "foot_site")
FOOT_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_contact")

# Altura objetivo
Z_TARGET = data.xpos[BOOM_ID][2] + 0.05

# Parámetros físicos
TAU_MAX = 3.73 * 26.9

# Temporización de la máquina de estados
STARTUP_LOCKOUT = 0.8
T_STANCE = 0.15
T_BLEND = 0.010

# Controlador de vuelo
KP_FLIGHT = np.diag([80.0, 80.0])
KD_FLIGHT = np.diag([12.0, 12.0])

# Controlador de apoyo
KP_HEIGHT = 200.0
_F_Z_PEAK = 400.0
F_Z_PEAK_MIN = 200.0
F_Z_PEAK_MAX = 600.0
OMEGA_YAW_DES = 0.0
F_Y_BASE = -3.0
K_RH = 10.0
F_Y_PEAK_MIN = -20.0
F_Y_PEAK_MAX = 0.0

# Anti-rebote de contacto
CONTACT_DEBOUNCE = 1
LIFTOFF_DEBOUNCE = 3
_contact_count = 0
_liftoff_count = 0

# Filtro para el encoder
ENC_ALPHA = np.exp(-2 * np.pi * 50 * model.opt.timestep)
_qpos_prev = np.zeros(2)
_qvel_filtered = np.zeros(2)

def update_encoder():
    global _qpos_prev, _qvel_filtered
    qpos = np.array([data.qpos[HIP_QADR], data.qpos[KNEE_QADR]])
    raw = (qpos - _qpos_prev) / model.opt.timestep
    _qvel_filtered = ENC_ALPHA * _qvel_filtered + (1 - ENC_ALPHA) * raw
    _qpos_prev = qpos.copy()
    return _qvel_filtered.copy()

# Configuración de referencia
_ref_data = mujoco.MjData(model)
mujoco.mj_resetData(model, _ref_data)
_ref_data.qpos[HIP_QADR] = 0.20
_ref_data.qpos[KNEE_QADR] = -0.8
mujoco.mj_forward(model, _ref_data)

# Posición de referencia del pie
hip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thigh")
R0 = np.eye(3)
hip_pos_world = _ref_data.xpos[hip_body_id]
foot_pos_world = _ref_data.site_xpos[FOOT_SITE]
hip_local = R0 @ hip_pos_world
foot_local = R0 @ foot_pos_world
PD_REF = (foot_local - hip_local)[1:3]

# Jacobiano del pie
def get_jacobian():
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, FOOT_SITE)
    return Jp[1:3, [HIP_DOF, KNEE_DOF]]

# Rotación de guiñada
def get_yaw_angle():
    yaw_jnt_adr = model.jnt_qposadr[yaw_jnt_id]
    return data.qpos[yaw_jnt_adr]

def yaw_rotation_matrix():
    theta = get_yaw_angle()
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def get_jacobian_local():
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, FOOT_SITE)
    R = yaw_rotation_matrix()
    Jp_local = R @ Jp
    return Jp_local[1:3, [HIP_DOF, KNEE_DOF]]

# Detección de contacto
def in_contact():
    total_fn = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == FOOT_GEOM_ID or c.geom2 == FOOT_GEOM_ID:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            total_fn += abs(force[0])
    return total_fn > 1.0, total_fn

# Seguimiento del punto más alto
_apex_z = 0.0
_apex_tracking = False

def reset_apex():
    global _apex_z, _apex_tracking
    _apex_z = data.xpos[BOOM_ID][2]
    _apex_tracking = True

def update_apex():
    global _apex_z
    if _apex_tracking:
        _apex_z = max(_apex_z, data.xpos[BOOM_ID][2])

def apex_to_fz():
    global _apex_tracking, _F_Z_PEAK
    _apex_tracking = False
    err = Z_TARGET - _apex_z
    new_peak = float(np.clip(_F_Z_PEAK + KP_HEIGHT * err, F_Z_PEAK_MIN, F_Z_PEAK_MAX))
    return new_peak

# Controladores
def flight_control(Jc, qvel_est):
    R = yaw_rotation_matrix()
    hip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thigh")
    
    hip_pos_world = data.xpos[hip_body_id]
    foot_pos_world = data.site_xpos[FOOT_SITE]
    
    hip_pos_local = R @ hip_pos_world
    foot_pos_local = R @ foot_pos_world
    
    p = foot_pos_local[1:3] - hip_pos_local[1:3]
    err = p - PD_REF
    p_dot = Jc @ qvel_est
    return -Jc.T @ (KP_FLIGHT @ err + KD_FLIGHT @ p_dot)

def bezier(t, peak):
    if t <= 0 or t >= T_STANCE:
        return 0.0
    s = t / T_STANCE
    return peak * (s**2 * (1 - s)**2) / 0.0625

def stance_control(t_in, Jc):
    omega = data.qvel[YAW_DOF]
    f_y = float(np.clip(F_Y_BASE - K_RH * (omega - OMEGA_YAW_DES),
                        F_Y_PEAK_MIN, F_Y_PEAK_MAX))
    t_y = max(t_in - T_STANCE * 0.30, 0.0)
    F = np.array([bezier(t_y, f_y), -bezier(t_in, _F_Z_PEAK)])
    return Jc.T @ F

# Máquina de estados
state = "FLIGHT"
t_state_start = 0.0
prev_state = state

def update_fsm():
    global state, t_state_start, _F_Z_PEAK
    global _contact_count, _liftoff_count
    
    contact, fz = in_contact()
    
    if data.time < STARTUP_LOCKOUT:
        return "FLIGHT", 0.0, fz
    
    if state == "FLIGHT":
        if contact:
            _contact_count += 1
            _liftoff_count = 0
        else:
            _contact_count = 0
        
        if _contact_count >= CONTACT_DEBOUNCE:
            _F_Z_PEAK = apex_to_fz()
            state = "STANCE"
            t_state_start = data.time
            _contact_count = 0
            _liftoff_count = 0
    
    elif state == "STANCE":
        if not contact:
            _liftoff_count += 1
            _contact_count = 0
        else:
            _liftoff_count = 0
        
        if _liftoff_count >= LIFTOFF_DEBOUNCE:
            reset_apex()
            state = "FLIGHT"
            t_state_start = data.time
            _contact_count = 0
            _liftoff_count = 0
    
    return state, data.time - t_state_start, fz

# Bucle principal
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat = [0.0, -3.0, 1.0]
            viewer.cam.distance = 20.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0
        
        for step in range(60_000):
            if not viewer.is_running():
                break
            
            mujoco.mj_step(model, data)
            
            update_apex()
            qvel_est = update_encoder()
            Jc = get_jacobian_local()
            cur_state, t_in, fz = update_fsm()
            
            if cur_state == "STANCE":
                tau_s = stance_control(t_in, Jc)
                tau_f = flight_control(Jc, qvel_est)
                alpha = min(t_in / T_BLEND, 1.0)
                tau = alpha * tau_s + (1 - alpha) * tau_f
            else:
                tau = flight_control(Jc, qvel_est)
            
            if not np.all(np.isfinite(tau)):
                tau = np.zeros(2)
            
            data.ctrl[0] = float(np.clip(tau[0], -TAU_MAX, TAU_MAX))
            data.ctrl[1] = float(np.clip(tau[1], -TAU_MAX, TAU_MAX))
            
            if step % 20 == 0:
                viewer.sync()
            
            prev_state = cur_state
finally:
    sys.exit(0)