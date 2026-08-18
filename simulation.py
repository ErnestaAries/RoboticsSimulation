import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Robot Arm Simulator")

# --- CSS: ẨN NÚT DEPLOY & TÙY CHỈNH GIAO DIỆN ---
custom_css = """
    <style>
        .stAppDeployButton {display: none;}
        div[data-testid="column"] button { width: 100%; border-radius: 5px; }
        .status-box {
            padding: 8px 15px; border-radius: 20px; font-weight: bold; 
            display: inline-block; float: right; border: 1px solid #ddd;
        }
        .status-red { color: #d62728; background-color: white; }
        .status-green { color: #2ca02c; background-color: white; }
        div[data-testid="stButton"] button p { font-size: 1.1rem !important; }
        div.row-widget.stRadio > div { flex-direction: column; gap: 0px; }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if 'num_joints' not in st.session_state: st.session_state.num_joints = 6
if 'full_dh_df' not in st.session_state:
    st.session_state.full_dh_df = pd.DataFrame([
        {'θ': 0.0, 'd': 80.0, 'a': 0.0, 'α': -90.0},
        {'θ': 0.0, 'd': 0.0, 'a': 120.0, 'α': 0.0},
        {'θ': 0.0, 'd': 0.0, 'a': 100.0, 'α': 0.0},
        {'θ': 0.0, 'd': 80.0, 'a': 0.0, 'α': -90.0},
        {'θ': 0.0, 'd': 0.0, 'a': 60.0, 'α': 90.0},
        {'θ': 0.0, 'd': 40.0, 'a': 0.0, 'α': 0.0},
    ])

def set_fk_preset(base, t1, t2, t3):
    st.session_state.fk_base = base
    st.session_state.fk_t1 = t1
    st.session_state.fk_t2 = t2
    if 'fk_t3' in st.session_state: st.session_state.fk_t3 = t3

def set_ik_preset(x, y, z):
    st.session_state.t_x = float(x)
    st.session_state.t_y = float(y)
    st.session_state.t_z = float(z)

# --- KINEMATICS FUNCTIONS ---
def forward_kinematics_2d(lengths, angles_deg):
    r, z = [0], [0]
    current_angle = 0
    for l, a in zip(lengths, angles_deg):
        current_angle += np.radians(a)
        r.append(r[-1] + l * np.cos(current_angle))
        z.append(z[-1] + l * np.sin(current_angle))
    return np.array(r), np.array(z)

# 1. HÀM GIẢI TÍCH (ANALYTIC IK) DÀNH RIÊNG CHO 2 LINKS
def analytic_ik_2d(target_r, target_z, l1, l2, elbow_mode):
    dist_sq = target_r**2 + target_z**2
    
    # Giới hạn tầm với để không bị lỗi toán học (căn số âm)
    max_reach_sq = (l1 + l2)**2
    min_reach_sq = (l1 - l2)**2
    if dist_sq > max_reach_sq:
        scale = np.sqrt(max_reach_sq / dist_sq)
        target_r, target_z = target_r * scale, target_z * scale
        dist_sq = max_reach_sq
    elif dist_sq < min_reach_sq:
        scale = np.sqrt(min_reach_sq / dist_sq)
        target_r, target_z = target_r * scale, target_z * scale
        dist_sq = min_reach_sq
        
    # Định lý hàm số Cosin
    c2 = (dist_sq - l1**2 - l2**2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)
    
    # ÉP NGHIỆM: s2 dương -> Gập xuống (Elbow down), s2 âm -> Gập lên (Elbow up)
    if elbow_mode == "Elbow down":
        s2 = np.sqrt(1 - c2**2)
    else:
        s2 = -np.sqrt(1 - c2**2)
        
    theta2 = np.arctan2(s2, c2)
    k1 = l1 + l2 * c2
    k2 = l2 * s2
    theta1 = np.arctan2(target_z, target_r) - np.arctan2(k2, k1)
    
    return np.degrees(np.array([theta1, theta2]))

# 2. HÀM DÒ DẪM (CCD IK) DÀNH CHO 3 LINKS TRỞ LÊN
def ccd_inverse_kinematics(target_r, target_z, lengths, max_iter=100, tolerance=0.1):
    angles = np.full(len(lengths), 0.01) # Mớm 0.01 để phá Singularity
    target = np.array([target_r, target_z])
    for _ in range(max_iter):
        for i in range(len(lengths)-1, -1, -1):
            r, z = forward_kinematics_2d(lengths, np.degrees(angles))
            ee_pos = np.array([r[-1], z[-1]])
            joint_pos = np.array([r[i], z[i]])
            v_ee = ee_pos - joint_pos
            v_target = target - joint_pos
            angle_ee = np.arctan2(v_ee[1], v_ee[0])
            angle_target = np.arctan2(v_target[1], v_target[0])
            angles[i] += (angle_target - angle_ee)
        r, z = forward_kinematics_2d(lengths, np.degrees(angles))
        if np.linalg.norm(np.array([r[-1], z[-1]]) - target) < tolerance:
            break
    return (np.degrees(angles) + 180) % 360 - 180

def dh_transform_matrix(theta, d, a, alpha):
    th = np.radians(theta)
    al = np.radians(alpha)
    return np.array([
        [np.cos(th), -np.sin(th)*np.cos(al),  np.sin(th)*np.sin(al), a*np.cos(th)],
        [np.sin(th),  np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
        [0,           np.sin(al),             np.cos(al),            d           ],
        [0,           0,                      0,                     1           ]
    ])

def draw_axes_3d(fig, T, scale=35):
    origin = T[:3, 3]
    vectors = [(T[:3, 0], 'red'), (T[:3, 1], 'lime'), (T[:3, 2], 'blue')]
    for vec, color in vectors:
        end_pt = origin + vec * scale
        fig.add_trace(go.Scatter3d(
            x=[origin[0], end_pt[0]], y=[origin[1], end_pt[1]], z=[origin[2], end_pt[2]],
            mode='lines', line=dict(color=color, width=5), hoverinfo='skip', showlegend=False
        ))
        fig.add_trace(go.Cone(
            x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]],
            u=[vec[0]], v=[vec[1]], w=[vec[2]],
            sizemode='absolute', sizeref=10, anchor='tail', 
            colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip'
        ))

# --- UI LAYOUT ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("Robot Arm Simulator")

col_ctrl, col_plot, col_out = st.columns([1, 2.5, 1])

with col_ctrl:
    st.subheader("Mode")
    mode = st.radio("", ["IK", "FK", "DH"], index=0, horizontal=True, label_visibility="collapsed", key="mode_radio")
    
    if mode in ["IK", "FK"]:
        st.write("**Links**")
        num_links_str = st.radio("", ["2 Links", "3 Links"], index=1, horizontal=True, label_visibility="collapsed", key="link_radio")
        num_links = 2 if "2" in num_links_str else 3
        
        st.write("**Link Lengths** (px)")
        l_cols = st.columns(3)
        l1 = l_cols[0].number_input("L1", min_value=1.0, max_value=500.0, value=120.0, key="l1")
        l2 = l_cols[1].number_input("L2", min_value=1.0, max_value=500.0, value=100.0, key="l2")
        l3 = l_cols[2].number_input("L3", min_value=1.0, max_value=500.0, value=80.0, key="l3") if num_links == 3 else 0
        lengths = [l1, l2, l3] if num_links == 3 else [l1, l2]
        max_reach = sum(lengths)
        
        if mode == "FK":
            st.write("**Joint Angles**")
            th_base = st.slider("Base (Pan)", -180, 180, 45, key="fk_base")
            th1 = st.slider("theta 1 (Tilt)", -180, 180, 45, key="fk_t1")
            th2 = st.slider("theta 2 (Tilt)", -180, 180, -30, key="fk_t2")
            th3 = st.slider("theta 3 (Tilt)", -180, 180, 10, key="fk_t3") if num_links == 3 else 0
            angles = [th1, th2, th3] if num_links == 3 else [th1, th2]
            
            st.write("**Presets**")
            p_col1, p_col2 = st.columns(2)
            p_col1.button("Inspect", on_click=set_fk_preset, args=(45, 45, -30, 10))
            p_col2.button("Reach", on_click=set_fk_preset, args=(0, 0, 0, 0))
            p_col1.button("Fold", on_click=set_fk_preset, args=(0, 120, -135, 90))
            p_col2.button("Reset", on_click=set_fk_preset, args=(0, 0, 0, 0))

        else: # IK
            st.write("**Target (Click +/- or Enter)**")
            t_cols1 = st.columns(2)
            t_x = t_cols1[0].number_input("X", min_value=-1000.0, max_value=1000.0, value=80.0, step=10.0, format="%.1f", key="t_x")
            t_y = t_cols1[1].number_input("Y", min_value=-1000.0, max_value=1000.0, value=150.0, step=10.0, format="%.1f", key="t_y")
            t_z = st.number_input("Z", min_value=-1000.0, max_value=1000.0, value=100.0, step=10.0, format="%.1f", key="t_z")
            
            st.write("") 
            if num_links == 2:
                elbow_mode = st.radio("Elbow Config", ["Elbow down", "Elbow up"], index=0, label_visibility="collapsed", key="elbow_mode")
                st.caption("2-link IK uses Analytic solving.")
            else:
                st.caption("3-link IK uses CCD solving.")
                elbow_mode = "Elbow down"
            
            target_dist = np.sqrt(t_x**2 + t_y**2 + t_z**2)
            
            st.write("**Presets**")
            p_col1, p_col2 = st.columns(2)
            p_col1.button("Inspect", on_click=set_ik_preset, args=(150, 150, 100))
            p_col2.button("Reach", on_click=set_ik_preset, args=(max_reach, 0, 0))
            p_col1.button("Fold", on_click=set_ik_preset, args=(30, 0, 30))
            p_col2.button("Reset", on_click=set_ik_preset, args=(max_reach/2, 0, max_reach/2))
            
            with header_col2:
                if target_dist > max_reach:
                    st.markdown('<div class="status-box status-red">🔴 Target outside reach</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-box status-green">🟢 Ready</div>', unsafe_allow_html=True)

    elif mode == "DH":
        st.write("**Joints**")
        j1, j2, j3 = st.columns([1.5, 1, 1.5])
        if j1.button("➖", use_container_width=True):
            if st.session_state.num_joints > 2: st.session_state.num_joints -= 1
        j2.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 5px;'>{st.session_state.num_joints}</div>", unsafe_allow_html=True)
        if j3.button("➕", use_container_width=True):
            if st.session_state.num_joints < 6: st.session_state.num_joints += 1
        
        st.write("**DH Parameters**")
        current_dh_df = st.session_state.full_dh_df.head(st.session_state.num_joints)
        edited_df = st.data_editor(current_dh_df, use_container_width=True, hide_index=False)
        st.session_state.full_dh_df.update(edited_df)
        
        st.write("**Joint Angles**")
        dh_angles = []
        for i in range(st.session_state.num_joints):
            dh_angles.append(st.slider(f"theta {i+1}", -180, 180, 0, key=f"dh_slider_{i}"))

# --- PLOTTING & CALCULATION ---
with col_plot:
    if mode in ["IK", "FK"]:
        st.subheader(f"{'Inverse' if mode=='IK' else 'Forward'} Kinematics (Full 3D Space)")
        
        if mode == "FK":
            base_rad = np.radians(th_base)
            r, z_pts = forward_kinematics_2d(lengths, angles)
        else: # IK
            base_rad = np.arctan2(t_y, t_x)
            t_r = np.sqrt(t_x**2 + t_y**2)
            
            # --- ĐIỀU HƯỚNG THUẬT TOÁN ---
            if len(lengths) == 2:
                # 2 Khớp: Dùng toán học giải tích (Chuẩn 100%)
                angles = analytic_ik_2d(t_r, t_z, lengths[0], lengths[1], elbow_mode)
            else:
                # 3 Khớp: Dùng thuật toán lặp dò dẫm CCD
                angles = ccd_inverse_kinematics(t_r, t_z, lengths)
                
            r, z_pts = forward_kinematics_2d(lengths, angles)
            th_base = np.degrees(base_rad)
        
        x_pts = r * np.cos(base_rad)
        y_pts = r * np.sin(base_rad)
        end_x, end_y, end_z = x_pts[-1], y_pts[-1], z_pts[-1]
        
        fig = go.Figure()
        
        # Cầu tầm với
        theta_cir = np.linspace(0, 2*np.pi, 60)
        c_cos = max_reach * np.cos(theta_cir)
        c_sin = max_reach * np.sin(theta_cir)
        z_zero = np.zeros_like(theta_cir)
        fig.add_trace(go.Scatter3d(x=c_cos, y=c_sin, z=z_zero, mode='lines', line=dict(color='lightblue', width=1, dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter3d(x=c_cos, y=z_zero, z=c_sin, mode='lines', line=dict(color='lightblue', width=1, dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter3d(x=z_zero, y=c_cos, z=c_sin, mode='lines', line=dict(color='lightblue', width=1, dash='dot'), showlegend=False))
        
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-20, 0], mode='lines', line=dict(color='gray', width=15), showlegend=False))
        
        for i in range(len(lengths)):
            fig.add_trace(go.Scatter3d(
                x=[x_pts[i], x_pts[i+1]], y=[y_pts[i], y_pts[i+1]], z=[z_pts[i], z_pts[i+1]], 
                mode='lines+markers', line=dict(color=colors[i], width=12), 
                marker=dict(size=8, color='white', line=dict(width=2, color='black')), showlegend=False
            ))
            
        if mode == "IK":
            t_color = 'royalblue' if target_dist <= max_reach else 'red'
            cross_len = 15
            fig.add_trace(go.Scatter3d(x=[t_x - cross_len, t_x + cross_len], y=[t_y, t_y], z=[t_z, t_z], mode='lines', line=dict(color=t_color, width=3), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[t_x, t_x], y=[t_y - cross_len, t_y + cross_len], z=[t_z, t_z], mode='lines', line=dict(color=t_color, width=3), showlegend=False))
            fig.add_trace(go.Scatter3d(x=[t_x, t_x], y=[t_y, t_y], z=[t_z - cross_len, t_z + cross_len], mode='lines', line=dict(color=t_color, width=3), showlegend=False))

        fig.update_layout(
            uirevision="constant", 
            scene=dict(
                xaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                yaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                zaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                aspectmode='cube'
            ), 
            margin=dict(l=0, r=0, t=0, b=0), height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    elif mode == "DH":
        st.subheader("DH Forward Kinematics (3D)")
        
        link_colors = ['#2c3e50', '#e74c3c', '#1abc9c', '#f39c12', '#9b59b6', '#34495e']
        T_matrices = [np.eye(4)]
        x_pts, y_pts, z_pts = [0], [0], [0]
        T_current = np.eye(4)
        
        for i in range(st.session_state.num_joints):
            row = edited_df.iloc[i]
            theta = row['θ'] + dh_angles[i]
            T_i = dh_transform_matrix(theta, row['d'], row['a'], row['α'])
            T_current = np.dot(T_current, T_i)
            T_matrices.append(T_current)
            x_pts.append(T_current[0, 3])
            y_pts.append(T_current[1, 3])
            z_pts.append(T_current[2, 3])
            
        end_transform = T_current
        end_x, end_y, end_z = T_current[0, 3], T_current[1, 3], T_current[2, 3]

        fig = go.Figure()
        
        for i in range(st.session_state.num_joints):
            fig.add_trace(go.Scatter3d(
                x=[x_pts[i], x_pts[i+1]], y=[y_pts[i], y_pts[i+1]], z=[z_pts[i], z_pts[i+1]], 
                mode='lines', line=dict(color=link_colors[i % len(link_colors)], width=18), showlegend=False
            ))
            
        fig.add_trace(go.Scatter3d(
            x=x_pts[:-1], y=y_pts[:-1], z=z_pts[:-1], 
            mode='markers', marker=dict(size=14, color='lightgray', line=dict(width=2, color='darkgray')), 
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[x_pts[-1]], y=[y_pts[-1]], z=[z_pts[-1]], 
            mode='markers', marker=dict(size=22, color='royalblue', line=dict(width=2, color='darkblue')), 
            showlegend=False
        ))
        
        for T in T_matrices:
            draw_axes_3d(fig, T, scale=35)

        fig.update_layout(
            uirevision="constant",
            scene=dict(
                xaxis=dict(range=[-350, 350], showbackground=False), 
                yaxis=dict(range=[-350, 350], showbackground=False), 
                zaxis=dict(range=[-350, 350], showbackground=False),
                aspectmode='cube'
            ), 
            margin=dict(l=0, r=0, t=0, b=0), height=600
        )
        st.plotly_chart(fig, use_container_width=True)

# --- OUTPUT METRICS ---
with col_out:
    if mode in ["IK", "FK"]:
        o_cols = st.columns(3)
        o_cols[0].metric("End X", f"{end_x:.1f}")
        o_cols[1].metric("End Y", f"{end_y:.1f}")
        o_cols[2].metric("End Z", f"{end_z:.1f}")
        
        if mode == "IK":
            error = np.sqrt((end_x - t_x)**2 + (end_y - t_y)**2 + (end_z - t_z)**2)
            st.metric("Error", f"{error:.1f}")
                
        st.write("---")
        st.write("**Joint Output**")
        st.write(f"Base Pan: &nbsp;&nbsp;&nbsp; **{th_base:.1f} deg**")
        for i, ang in enumerate(angles):
            st.write(f"theta {i+1} (Tilt): **{ang:.1f} deg**")
            
    elif mode == "DH":
        o_cols1 = st.columns(2)
        o_cols1[0].metric("End X", f"{end_x:.1f}")
        o_cols1[1].metric("End Y", f"{end_y:.1f}")
        o_cols2 = st.columns(2)
        o_cols2[0].metric("End Z", f"{end_z:.1f}")
        o_cols2[1].metric("Error", "0.0")
        
        st.write("---")
        st.write("**Joint Output**")
        for i in range(st.session_state.num_joints):
            st.write(f"theta {i+1}: &nbsp;&nbsp;&nbsp;&nbsp; **{dh_angles[i]:.1f} deg**")
        
        st.write("---")
        st.write("**End-Effector Transform**")
        st.dataframe(pd.DataFrame(end_transform).round(3))