import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Robotics Simulation Hub")

# --- CSS TÙY CHỈNH ---
custom_css = """
    <style>
        .stAppDeployButton {display: none;}
        div[data-testid="column"] button { width: 100%; border-radius: 5px; }
        .status-box { padding: 8px 15px; border-radius: 20px; font-weight: bold; display: inline-block; float: right; border: 1px solid #ddd; }
        .status-red { color: #d62728; background-color: white; }
        .status-green { color: #2ca02c; background-color: white; }
        div[data-testid="stButton"] button p { font-size: 1.1rem !important; }
        div.row-widget.stRadio > div { flex-direction: column; gap: 0px; }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =====================================================================
# SIDEBAR: CHỌN LOẠI ROBOT
# =====================================================================
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 4rem; margin-bottom: 0;'>🤖</h1>", unsafe_allow_html=True)
    st.title("Robot Selector")
    robot_type = st.radio(
        "Chọn nền tảng mô phỏng:",
        ["Cánh tay nối tiếp (Articulated)", 
         "Robot Tuyến tính (Cartesian)", 
         "SCARA Robot", 
         "Delta Robot"]
    )
    st.markdown("---")
    st.caption("Developed for Robotics Education")


# =====================================================================
# QUẢN LÝ TRẠNG THÁI VÀ CÁC HÀM TOÁN HỌC DÙNG CHUNG
# =====================================================================
if 'num_joints' not in st.session_state: st.session_state.num_joints = 3
if 'fk_base' not in st.session_state: st.session_state.fk_base = 45
if 'fk_t1' not in st.session_state: st.session_state.fk_t1 = 45
if 'fk_t2' not in st.session_state: st.session_state.fk_t2 = -30
if 'fk_t3' not in st.session_state: st.session_state.fk_t3 = 10
if 't_x' not in st.session_state: st.session_state.t_x = 80.0
if 't_y' not in st.session_state: st.session_state.t_y = 150.0
if 't_z' not in st.session_state: st.session_state.t_z = 100.0

if 'full_dh_df' not in st.session_state:
    st.session_state.full_dh_df = pd.DataFrame([
        {'θ': 0.0, 'd': 80.0, 'a': 0.0, 'α': -90.0},
        {'θ': 0.0, 'd': 0.0, 'a': 120.0, 'α': 0.0},
        {'θ': 0.0, 'd': 0.0, 'a': 100.0, 'α': 0.0},
        {'θ': 0.0, 'd': 80.0, 'a': 0.0, 'α': -90.0},
        {'θ': 0.0, 'd': 0.0, 'a': 60.0, 'α': 90.0},
        {'θ': 0.0, 'd': 40.0, 'a': 0.0, 'α': 0.0},
    ])

if 'student_formulas_dh' not in st.session_state: st.session_state.student_formulas_dh = [[["1", "0", "0", "0"], ["0", "1", "0", "0"], ["0", "0", "1", "0"], ["0", "0", "0", "1"]] for _ in range(6)]
if 'dh_step_unlocked' not in st.session_state: st.session_state.dh_step_unlocked = 0
if 'ik_formulas' not in st.session_state: st.session_state.ik_formulas = [""] * 6
if 'ik_calculated_thetas' not in st.session_state: st.session_state.ik_calculated_thetas = None
if 'ik_inter_df' not in st.session_state:
    st.session_state.ik_inter_df = pd.DataFrame([
        {"Tên biến": "r_sq", "Công thức": "X^2 + Y^2"}, {"Tên biến": "s", "Công thức": "d1 - Z"}, {"Tên biến": "D", "Công thức": "(r_sq + s^2 - a2^2 - a3^2) / (2 * a2 * a3)"}
    ])

# STATE MỚI CHO TÍNH NĂNG "KHÁM PHÁ D-H"
if 'exp_step' not in st.session_state: st.session_state.exp_step = 0
if 'exp_rand_target' not in st.session_state: st.session_state.exp_rand_target = None

def trans_z(d): return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]])
def rot_z(th): th_r = np.radians(th); return np.array([[np.cos(th_r),-np.sin(th_r),0,0],[np.sin(th_r),np.cos(th_r),0,0],[0,0,1,0],[0,0,0,1]])
def trans_x(a): return np.array([[1,0,0,a],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
def rot_x(al): al_r = np.radians(al); return np.array([[1,0,0,0],[0,np.cos(al_r),-np.sin(al_r),0],[0,np.sin(al_r),np.cos(al_r),0],[0,0,0,1]])

def set_fk_preset(base, t1, t2, t3):
    st.session_state.fk_base = base; st.session_state.fk_t1 = t1; st.session_state.fk_t2 = t2
    if st.session_state.num_links == 3: st.session_state.fk_t3 = t3
def set_ik_preset(x, y, z): st.session_state.t_x = float(x); st.session_state.t_y = float(y); st.session_state.t_z = float(z)

def forward_kinematics_2d(lengths, angles_deg):
    r, z = [0], [0]
    current_angle = 0
    for l, a in zip(lengths, angles_deg):
        current_angle += np.radians(a)
        r.append(r[-1] + l * np.cos(current_angle))
        z.append(z[-1] + l * np.sin(current_angle))
    return np.array(r), np.array(z)

def analytic_ik_2d(target_r, target_z, l1, l2, elbow_mode):
    dist_sq = target_r**2 + target_z**2
    max_reach_sq, min_reach_sq = (l1 + l2)**2, (l1 - l2)**2
    if dist_sq > max_reach_sq:
        scale = np.sqrt(max_reach_sq / dist_sq)
        target_r, target_z, dist_sq = target_r * scale, target_z * scale, max_reach_sq
    elif dist_sq < min_reach_sq:
        scale = np.sqrt(min_reach_sq / dist_sq)
        target_r, target_z, dist_sq = target_r * scale, target_z * scale, min_reach_sq
    c2 = np.clip((dist_sq - l1**2 - l2**2) / (2 * l1 * l2), -1.0, 1.0)
    s2 = np.sqrt(1 - c2**2) if elbow_mode == "Elbow down" else -np.sqrt(1 - c2**2)
    theta2 = np.arctan2(s2, c2)
    k1, k2 = l1 + l2 * c2, l2 * s2
    theta1 = np.arctan2(target_z, target_r) - np.arctan2(k2, k1)
    return np.degrees(np.array([theta1, theta2]))

def ccd_inverse_kinematics(target_r, target_z, lengths, max_iter=100, tolerance=0.1):
    angles = np.full(len(lengths), 0.01)
    target = np.array([target_r, target_z])
    for _ in range(max_iter):
        for i in range(len(lengths)-1, -1, -1):
            r, z = forward_kinematics_2d(lengths, np.degrees(angles))
            v_ee = np.array([r[-1], z[-1]]) - np.array([r[i], z[i]])
            v_target = target - np.array([r[i], z[i]])
            angles[i] += np.arctan2(v_target[1], v_target[0]) - np.arctan2(v_ee[1], v_ee[0])
        r, z = forward_kinematics_2d(lengths, np.degrees(angles))
        if np.linalg.norm(np.array([r[-1], z[-1]]) - target) < tolerance: break
    return (np.degrees(angles) + 180) % 360 - 180

def dh_transform_matrix(theta, d, a, alpha):
    th, al = np.radians(theta), np.radians(alpha)
    return np.array([[np.cos(th), -np.sin(th)*np.cos(al), np.sin(th)*np.sin(al), a*np.cos(th)],
                     [np.sin(th), np.cos(th)*np.cos(al), -np.cos(th)*np.sin(al), a*np.sin(th)],
                     [0, np.sin(al), np.cos(al), d], [0, 0, 0, 1]])

def evaluate_student_formula(expr_str, theta_val, d_val, a_val, alpha_val):
    if not expr_str: return 0.0
    safe_dict = {'cos': lambda x: np.cos(np.radians(x)), 'sin': lambda x: np.sin(np.radians(x)), 'theta': theta_val, 'θ': theta_val, 'd': d_val, 'a': a_val, 'alpha': alpha_val, 'α': alpha_val}
    try: return float(eval(str(expr_str).replace('^', '**'), {"__builtins__": {}}, safe_dict))
    except: return None

def evaluate_ik_formula(formulas, tx, ty, tz, dh_df, num_j, inter_df):
    safe_dict = {
        'X': tx, 'Y': ty, 'Z': tz, 'x': tx, 'y': ty, 'z': tz,
        'cos': lambda a: np.cos(np.radians(a)), 'sin': lambda a: np.sin(np.radians(a)),
        'acos': lambda v: np.degrees(np.arccos(np.clip(v, -1.0, 1.0))), 
        'asin': lambda v: np.degrees(np.arcsin(np.clip(v, -1.0, 1.0))),
        'atan2': lambda y, x: np.degrees(np.arctan2(y, x)), 'sqrt': np.sqrt, 'pi': np.pi
    }
    for i in range(len(dh_df)):
        safe_dict[f'a{i+1}'] = dh_df.iloc[i]['a']; safe_dict[f'd{i+1}'] = dh_df.iloc[i]['d']; safe_dict[f'alpha{i+1}'] = dh_df.iloc[i]['α']
    
    for idx, row in inter_df.iterrows():
        v_name, v_form = str(row['Tên biến']).strip(), str(row['Công thức']).strip()
        if v_name and v_form:
            try: safe_dict[v_name] = float(eval(v_form.replace('^', '**'), {"__builtins__": {}}, safe_dict))
            except Exception as e: return None, f"Lỗi ở biến phụ '{v_name}': {str(e)}"

    results, unsolved = [0.0] * num_j, list(range(num_j))
    for _ in range(num_j):
        for i in unsolved.copy():
            f = formulas[i]
            if not f.strip(): unsolved.remove(i)
            else:
                try:
                    val = float(eval(str(f).replace('^', '**'), {"__builtins__": {}}, safe_dict))
                    results[i], safe_dict[f'theta{i+1}'] = val, val
                    unsolved.remove(i)
                except Exception: pass 
                    
    if unsolved:
        first_fail = unsolved[0]
        try: eval(str(formulas[first_fail]).replace('^', '**'), {"__builtins__": {}}, safe_dict)
        except Exception as e: return None, f"Ô theta {first_fail+1}: {str(e)}"
    return results, "OK"

def draw_axes_3d(fig, T, scale=35, opacity=1.0):
    origin = T[:3, 3]
    for vec, color in [(T[:3, 0], 'red'), (T[:3, 1], 'lime'), (T[:3, 2], 'blue')]:
        end_pt = origin + vec * scale
        fig.add_trace(go.Scatter3d(x=[origin[0], end_pt[0]], y=[origin[1], end_pt[1]], z=[origin[2], end_pt[2]], mode='lines', line=dict(color=color, width=5), opacity=opacity, hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Cone(x=[end_pt[0]], y=[end_pt[1]], z=[end_pt[2]], u=[vec[0]], v=[vec[1]], w=[vec[2]], sizemode='absolute', sizeref=10, anchor='tail', colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity, hoverinfo='skip'))

def add_cylinder(fig, T, radius=8, height=24, color='#d3d3d3', opacity=1.0):
    z_vals = np.linspace(-height/2, height/2, 2)
    theta = np.linspace(0, 2*np.pi, 20)
    th_g, z_g = np.meshgrid(theta, z_vals)
    x_g, y_g = radius * np.cos(th_g), radius * np.sin(th_g)
    
    X, Y, Z = np.zeros_like(x_g), np.zeros_like(y_g), np.zeros_like(z_g)
    for i in range(x_g.shape[0]):
        for j in range(x_g.shape[1]):
            pt_w = np.dot(T, np.array([x_g[i,j], y_g[i,j], z_g[i,j], 1]))
            X[i,j], Y[i,j], Z[i,j] = pt_w[:3]
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip', opacity=opacity))
    
    r_g, th2_g = np.meshgrid(np.linspace(0, radius, 2), theta)
    xc, yc = r_g * np.cos(th2_g), r_g * np.sin(th2_g)
    for z_off in [-height/2, height/2]:
        Xc, Yc, Zc = np.zeros_like(xc), np.zeros_like(yc), np.zeros_like(xc)
        for i in range(xc.shape[0]):
            for j in range(xc.shape[1]):
                pt_w = np.dot(T, np.array([xc[i,j], yc[i,j], z_off, 1]))
                Xc[i,j], Yc[i,j], Zc[i,j] = pt_w[:3]
        fig.add_trace(go.Surface(x=Xc, y=Yc, z=Zc, colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip', opacity=opacity))


# =====================================================================
# MODULE 1: CÁNH TAY NỐI TIẾP (BẢN FULL OPTION ĐÃ FIX LỖI UI)
# =====================================================================
if robot_type == "Cánh tay nối tiếp (Articulated)":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1: st.title("Robot Arm Simulator")

    col_ctrl, col_plot, col_out = st.columns([1.5, 2.0, 0.8])

    with col_ctrl:
        st.subheader("Mode")
        mode = st.radio("", ["IK", "FK", "DH"], index=2, horizontal=True, label_visibility="collapsed", key="mode_radio")
        
        if mode in ["IK", "FK"]:
            st.write("**Links**")
            num_links = 2 if "2" in st.radio("", ["2 Links", "3 Links"], index=1, horizontal=True, label_visibility="collapsed") else 3
            st.session_state.num_links = num_links
            l_cols = st.columns(3)
            lengths = [l_cols[0].number_input("L1", 1.0, 500.0, 120.0), l_cols[1].number_input("L2", 1.0, 500.0, 100.0)]
            if num_links == 3: lengths.append(l_cols[2].number_input("L3", 1.0, 500.0, 80.0))
            max_reach = sum(lengths)
            
            if mode == "FK":
                th_base = st.slider("Base (Pan)", -180, 180, key="fk_base")
                angles = [st.slider("theta 1", -180, 180, key="fk_t1"), st.slider("theta 2", -180, 180, key="fk_t2")]
                if num_links == 3: angles.append(st.slider("theta 3", -180, 180, key="fk_t3"))
            else:
                t_cols = st.columns(2)
                t_x, t_y = t_cols[0].number_input("X", -1000.0, 1000.0, step=10.0, key="t_x"), t_cols[1].number_input("Y", -1000.0, 1000.0, step=10.0, key="t_y")
                t_z = st.number_input("Z", -1000.0, 1000.0, step=10.0, key="t_z")
                elbow_mode = st.radio("Elbow Config", ["Elbow down", "Elbow up"], index=0, label_visibility="collapsed") if num_links == 2 else "Elbow down"
                with header_col2:
                    if np.sqrt(t_x**2 + t_y**2 + t_z**2) > max_reach: st.markdown('<div class="status-box status-red">🔴 Target outside reach</div>', unsafe_allow_html=True)
                    else: st.markdown('<div class="status-box status-green">🟢 Ready</div>', unsafe_allow_html=True)

        elif mode == "DH":
            practice_mode = st.toggle("🎓 Bật Practice Mode")
            
            st.write("**Joints**")
            j1, j2, j3 = st.columns([1.5, 1, 1.5])
            if j1.button("➖", use_container_width=True) and st.session_state.num_joints > 2: st.session_state.num_joints -= 1
            j2.markdown(f"<div style='text-align: center; font-size: 20px; font-weight: bold; margin-top: 5px;'>{st.session_state.num_joints}</div>", unsafe_allow_html=True)
            if j3.button("➕", use_container_width=True) and st.session_state.num_joints < 6: st.session_state.num_joints += 1
            
            st.write("**DH Parameters**")
            current_dh_df = st.session_state.full_dh_df.head(st.session_state.num_joints)
            st.session_state.full_dh_df.update(st.data_editor(current_dh_df, use_container_width=True, hide_index=False))
            
            if not practice_mode:
                st.write("**Joint Angles**")
                dh_angles = [st.slider(f"theta {i+1}", -180, 180, 0, key=f"dh_slider_{i}") for i in range(st.session_state.num_joints)]
            else:
                prac_type = st.radio("Loại bài tập:", ["📚 FK: Từng bước", "🎯 IK: Giải tích", "🔍 Khám phá D-H"], horizontal=True)
                
                # --- UI CHO THỰC HÀNH TỪNG BƯỚC FK ---
                if prac_type == "📚 FK: Từng bước":
                    st.info("💡 Nhập ma trận biến đổi. Có thể dùng số `0`, `1` hoặc công thức `cos(theta)`, `-sin(theta)*cos(alpha)`.")
                    current_step = st.session_state.dh_step_unlocked
                    is_success = False
                    
                    gt_matrices, T_current_gt = [], np.eye(4)
                    for i in range(st.session_state.num_joints):
                        row = current_dh_df.iloc[i]
                        T_i = dh_transform_matrix(row['θ'] + 0, row['d'], row['a'], row['α']) 
                        T_current_gt = np.dot(T_current_gt, T_i); gt_matrices.append(T_i)
                        
                    if current_step < st.session_state.num_joints:
                        st.write(f"### Nhập ma trận $T_{current_step}^{current_step+1}$")
                        with st.form(key=f"form_step_{current_step}"):
                            edited_matrix_df = st.data_editor(pd.DataFrame(st.session_state.student_formulas_dh[current_step], columns=['Col 1', 'Col 2', 'Col 3', 'Col 4'], dtype=str), use_container_width=True, hide_index=True)
                            if st.form_submit_button(f"Kiểm tra Khớp {current_step+1}"):
                                st.session_state.student_formulas_dh[current_step] = edited_matrix_df.values.tolist()
                                row = current_dh_df.iloc[current_step]
                                student_eval_matrix, has_syntax_error = np.zeros((4,4)), False
                                for r in range(4):
                                    for c in range(4):
                                        val = evaluate_student_formula(st.session_state.student_formulas_dh[current_step][r][c], row['θ'], row['d'], row['a'], row['α'])
                                        if val is None: has_syntax_error = True; break
                                        student_eval_matrix[r, c] = val
                                if has_syntax_error: st.error("❌ Cú pháp sai!")
                                elif np.allclose(student_eval_matrix, gt_matrices[current_step], atol=0.05):
                                    st.success("✅ Chính xác!"); st.session_state.dh_step_unlocked += 1; st.rerun()
                                else: st.error("Sai rồi!")
                        with st.expander("Gợi ý công thức"): st.latex(r"T = \begin{bmatrix} \cos(\theta) & -\sin(\theta)\cos(\alpha) & \sin(\theta)\sin(\alpha) & a\cos(\theta) \\ \sin(\theta) & \cos(\theta)\cos(\alpha) & -\cos(\theta)\sin(\alpha) & a\sin(\theta) \\ 0 & \sin(\alpha) & \cos(\alpha) & d \\ 0 & 0 & 0 & 1 \end{bmatrix}")
                    else:
                        st.success("🎉 Bạn đã giải đúng toàn bộ!")
                        if st.button("Làm lại từ đầu"): st.session_state.dh_step_unlocked = 0; st.rerun()

                # --- UI CHO THỰC HÀNH GIẢI TÍCH IK ---
                elif prac_type == "🎯 IK: Giải tích":
                    if st.session_state.num_joints > 3:
                        st.warning("⚠️ IK Giải tích chỉ hỗ trợ tối đa 3 bậc tự do.")
                    else:
                        t_cols = st.columns(3)
                        t_x_ik = t_cols[0].number_input("Target X", -1000.0, 1000.0, 150.0, step=10.0)
                        t_y_ik = t_cols[1].number_input("Target Y", -1000.0, 1000.0, 150.0, step=10.0)
                        t_z_ik = t_cols[2].number_input("Target Z", -1000.0, 1000.0, 100.0, step=10.0)
                        
                        st.write("### 1. Bảng Biến Trung Gian")
                        edited_inter_df = st.data_editor(st.session_state.ik_inter_df, num_rows="dynamic", use_container_width=True)
                        st.session_state.ik_inter_df = edited_inter_df
                        
                        st.write("### 2. Nhập Nghiệm Các Khớp")
                        with st.form("ik_form"):
                            for i in range(st.session_state.num_joints):
                                st.session_state.ik_formulas[i] = st.text_input(f"Nhập nghiệm theta {i+1} (Độ):", value=st.session_state.ik_formulas[i])
                            if st.form_submit_button("Kiểm tra IK"):
                                res, err = evaluate_ik_formula(st.session_state.ik_formulas, t_x_ik, t_y_ik, t_z_ik, current_dh_df, st.session_state.num_joints, st.session_state.ik_inter_df)
                                if res is None:
                                    st.error(f"❌ Lỗi: {err}"); st.session_state.ik_calculated_thetas = None
                                else:
                                    st.session_state.ik_calculated_thetas = res
                                    T_test = np.eye(4)
                                    for i in range(st.session_state.num_joints):
                                        row = current_dh_df.iloc[i]
                                        T_test = np.dot(T_test, dh_transform_matrix(row['θ'] + res[i], row['d'], row['a'], row['α']))
                                    dist = np.linalg.norm(T_test[:3, 3] - np.array([t_x_ik, t_y_ik, t_z_ik]))
                                    if dist < 1.0: st.success(f"🎉 Rất chính xác! Sai số chỉ {dist:.2f} mm.")
                                    else: st.error(f"❌ Sai số: {dist:.1f} mm. Hãy kiểm tra lại công thức định lý Cosin hoặc Atan2!")

                # ================= ĐÃ LÀM LẠI: 2 TÙY CHỌN D-H HOÀN TOÀN MỚI =================
                elif prac_type == "🔍 Khám phá D-H":
                    exp_mode = st.radio("Chọn chế độ:", ["🎬 1. Phân tích D-H (Trực quan hóa)", "🎲 2. Thử thách giải đố (Blind Test)"], horizontal=True)
                    st.write("---")
                    
                    if exp_mode == "🎬 1. Phân tích D-H (Trực quan hóa)":
                        curr = st.session_state.exp_step
                        if curr < st.session_state.num_joints:
                            gt_row = current_dh_df.iloc[curr]
                            st.markdown(f"### Đang phân tích: Khớp {curr} ➔ Khớp {curr+1}")
                            st.info(f"💡 Dựa vào bảng D-H bạn đã nhập ở cột bên trái, thông số thật của khớp này là: **d = {gt_row['d']}, θ = {gt_row['θ']}, a = {gt_row['a']}, α = {gt_row['α']}**.")
                            st.write("👉 **Nhiệm vụ:** Hãy tự tay kéo 4 thanh trượt dưới đây từ $0$ tiến về đúng các giá trị trên để quan sát trục tọa độ biến đổi như thế nào!")
                            
                            dh_cols = st.columns(4)
                            # GIẢI QUYẾT LAG BẰNG KEY ĐỘC LẬP
                            val_d = dh_cols[0].slider(f"1. Tịnh tiến Z (d)", -200.0, 200.0, 0.0, step=10.0, key=f"d_vis_{curr}")
                            val_theta = dh_cols[1].slider(f"2. Xoay Z (θ)", -180.0, 180.0, 0.0, step=15.0, key=f"th_vis_{curr}")
                            val_a = dh_cols[2].slider(f"3. Tịnh tiến X (a)", -200.0, 200.0, 0.0, step=10.0, key=f"a_vis_{curr}")
                            val_alpha = dh_cols[3].slider(f"4. Xoay X (α)", -180.0, 180.0, 0.0, step=15.0, key=f"al_vis_{curr}")
                            
                            T_stud = dh_transform_matrix(val_theta, val_d, val_a, val_alpha)
                            T_gt = dh_transform_matrix(gt_row['θ'], gt_row['d'], gt_row['a'], gt_row['α'])
                            
                            if np.allclose(T_stud, T_gt, atol=0.05):
                                st.success("✅ Trục tọa độ đã trùng khớp hoàn toàn!")
                                if st.button("Chốt & Phân tích khớp tiếp theo ➔"):
                                    st.session_state.exp_step += 1; st.rerun()
                        else:
                            st.success("🎉 Bạn đã phân tích xong toàn bộ Robot!")
                            if st.button("Làm lại từ đầu"): st.session_state.exp_step = 0; st.rerun()

                    elif exp_mode == "🎲 2. Thử thách giải đố (Blind Test)":
                        st.info("💡 Hệ thống đã giấu bảng D-H và tự tạo một Hệ Tọa Độ Mục Tiêu (Bóng mờ) ngẫu nhiên. Nhiệm vụ của bạn là kéo 4 thanh trượt để dò ra thông số D-H đó!")
                        
                        if st.button("🔄 Tạo thử thách mới") or st.session_state.exp_rand_target is None:
                            st.session_state.exp_rand_target = {
                                'd': float(np.random.choice([-100, -50, 0, 50, 100])),
                                'theta': float(np.random.choice([-90, -45, 0, 45, 90, 180])),
                                'a': float(np.random.choice([-100, -50, 0, 50, 100])),
                                'alpha': float(np.random.choice([-90, 0, 90]))
                            }
                            st.rerun()

                        dh_cols = st.columns(4)
                        r_d = dh_cols[0].slider(f"1. Tịnh tiến Z (d)", -200.0, 200.0, 0.0, step=10.0, key="quiz_d")
                        r_th = dh_cols[1].slider(f"2. Xoay Z (θ)", -180.0, 180.0, 0.0, step=15.0, key="quiz_th")
                        r_a = dh_cols[2].slider(f"3. Tịnh tiến X (a)", -200.0, 200.0, 0.0, step=10.0, key="quiz_a")
                        r_al = dh_cols[3].slider(f"4. Xoay X (α)", -180.0, 180.0, 0.0, step=15.0, key="quiz_al")

                        if st.button("✅ Trả lời!"):
                            target = st.session_state.exp_rand_target
                            T_stud = dh_transform_matrix(r_th, r_d, r_a, r_al)
                            T_tar = dh_transform_matrix(target['theta'], target['d'], target['a'], target['alpha'])
                            if np.allclose(T_stud, T_tar, atol=0.05):
                                st.success("🎉 Quá xuất sắc! Bạn đã hiểu hoàn toàn bản chất của D-H.")
                            else:
                                st.error("❌ Chưa khớp rồi! Gợi ý: Hãy quan sát hướng của trục Z (Màu xanh) và trục X (Màu đỏ).")
                                with st.expander("👀 Xem đáp án"):
                                    st.write(f"d = {target['d']}, θ = {target['theta']}, a = {target['a']}, α = {target['alpha']}")

    # ------------------ KHU VỰC CHỈ DÀNH ĐỂ VẼ 3D ------------------
    with col_plot:
        fig = go.Figure()

        if mode in ["IK", "FK"]:
            if mode == "FK": base_rad, angles_res = np.radians(th_base), angles
            else:
                base_rad, t_r = np.arctan2(t_y, t_x), np.sqrt(t_x**2 + t_y**2)
                angles_res = analytic_ik_2d(t_r, t_z, lengths[0], lengths[1], elbow_mode) if len(lengths) == 2 else ccd_inverse_kinematics(t_r, t_z, lengths)
                th_base, angles = np.degrees(base_rad), angles_res
            r, z_pts = forward_kinematics_2d(lengths, angles_res)
            x_pts, y_pts = r * np.cos(base_rad), r * np.sin(base_rad)
            end_x, end_y, end_z = x_pts[-1], y_pts[-1], z_pts[-1]
            
            colors = ['#1f77b4', '#d62728', '#2ca02c']
            fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[-20, 0], mode='lines', line=dict(color='gray', width=15), showlegend=False))
            for i in range(len(lengths)):
                fig.add_trace(go.Scatter3d(x=[x_pts[i], x_pts[i+1]], y=[y_pts[i], y_pts[i+1]], z=[z_pts[i], z_pts[i+1]], mode='lines+markers', line=dict(color=colors[i], width=12), marker=dict(size=8, color='white'), showlegend=False))
            if mode == "IK":
                t_color = 'royalblue' if np.sqrt(t_x**2 + t_y**2 + t_z**2) <= max_reach else 'red'
                for x, y, z in [([t_x - 15, t_x + 15], [t_y, t_y], [t_z, t_z]), ([t_x, t_x], [t_y - 15, t_y + 15], [t_z, t_z]), ([t_x, t_x], [t_y, t_y], [t_z - 15, t_z + 15])]:
                    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=t_color, width=3), showlegend=False))

        elif mode == "DH":
            link_colors = ['#2c3e50', '#e74c3c', '#1abc9c', '#f39c12', '#9b59b6', '#34495e']
            
            if not practice_mode:
                T_matrices, x_pts, y_pts, z_pts, T_current = [np.eye(4)], [0], [0], [0], np.eye(4)
                for i in range(st.session_state.num_joints):
                    row = current_dh_df.iloc[i]
                    T_i = dh_transform_matrix(row['θ'] + dh_angles[i], row['d'], row['a'], row['α'])
                    T_current = np.dot(T_current, T_i)
                    T_matrices.append(T_current); x_pts.append(T_current[0, 3]); y_pts.append(T_current[1, 3]); z_pts.append(T_current[2, 3])
                end_x, end_y, end_z = T_current[0, 3], T_current[1, 3], T_current[2, 3]
                
                for i in range(st.session_state.num_joints):
                    fig.add_trace(go.Scatter3d(x=[x_pts[i], x_pts[i+1]], y=[y_pts[i], y_pts[i+1]], z=[z_pts[i], z_pts[i+1]], mode='lines', line=dict(color=link_colors[i % len(link_colors)], width=18), showlegend=False))
                fig.add_trace(go.Scatter3d(x=[x_pts[-1]], y=[y_pts[-1]], z=[z_pts[-1]], mode='markers', marker=dict(size=22, color='royalblue', line=dict(width=2, color='darkblue')), showlegend=False))
                for i, T in enumerate(T_matrices[:-1]): add_cylinder(fig, T); draw_axes_3d(fig, T, scale=35)
                draw_axes_3d(fig, T_matrices[-1], scale=35)

            else:
                if prac_type == "📚 FK: Từng bước":
                    T_draw, x_pts, y_pts, z_pts = np.eye(4), [0], [0], [0]
                    add_cylinder(fig, T_draw); draw_axes_3d(fig, T_draw, scale=35)
                    for i in range(st.session_state.dh_step_unlocked):
                        row = current_dh_df.iloc[i]
                        T_draw = np.dot(T_draw, dh_transform_matrix(row['θ'] + 0, row['d'], row['a'], row['α'])) 
                        x_pts.append(T_draw[0, 3]); y_pts.append(T_draw[1, 3]); z_pts.append(T_draw[2, 3])
                        fig.add_trace(go.Scatter3d(x=[x_pts[-2], x_pts[-1]], y=[y_pts[-2], y_pts[-1]], z=[z_pts[-2], z_pts[-1]], mode='lines', line=dict(color=link_colors[i % len(link_colors)], width=18), showlegend=False))
                        add_cylinder(fig, T_draw); draw_axes_3d(fig, T_draw, scale=35)
                    end_x, end_y, end_z = (x_pts[-1], y_pts[-1], z_pts[-1]) if st.session_state.dh_step_unlocked > 0 else (0,0,0)

                elif prac_type == "🎯 IK: Giải tích":
                    if st.session_state.num_joints > 3:
                        T_draw = np.eye(4); add_cylinder(fig, T_draw); draw_axes_3d(fig, T_draw, scale=35); end_x, end_y, end_z = 0, 0, 0
                    else:
                        T_draw, x_pts, y_pts, z_pts = np.eye(4), [0], [0], [0]
                        add_cylinder(fig, T_draw)
                        for x, y, z in [([t_x_ik - 15, t_x_ik + 15], [t_y_ik, t_y_ik], [t_z_ik, t_z_ik]), ([t_x_ik, t_x_ik], [t_y_ik - 15, t_y_ik + 15], [t_z_ik, t_z_ik]), ([t_x_ik, t_x_ik], [t_y_ik, t_y_ik], [t_z_ik - 15, t_z_ik + 15])]:
                            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red', width=3), showlegend=False))
                        if st.session_state.ik_calculated_thetas is not None:
                            thetas_render = st.session_state.ik_calculated_thetas
                            for i in range(st.session_state.num_joints):
                                row = current_dh_df.iloc[i]
                                T_draw = np.dot(T_draw, dh_transform_matrix(row['θ'] + thetas_render[i], row['d'], row['a'], row['α']))
                                x_pts.append(T_draw[0, 3]); y_pts.append(T_draw[1, 3]); z_pts.append(T_draw[2, 3])
                                fig.add_trace(go.Scatter3d(x=[x_pts[-2], x_pts[-1]], y=[y_pts[-2], y_pts[-1]], z=[z_pts[-2], z_pts[-1]], mode='lines', line=dict(color=link_colors[i % len(link_colors)], width=18), showlegend=False))
                                add_cylinder(fig, T_draw)
                            fig.add_trace(go.Scatter3d(x=[x_pts[-1]], y=[y_pts[-1]], z=[z_pts[-1]], mode='markers', marker=dict(size=22, color='royalblue', line=dict(width=2, color='darkblue')), showlegend=False))
                            end_x, end_y, end_z = x_pts[-1], y_pts[-1], z_pts[-1]
                        else: end_x, end_y, end_z = 0, 0, 0

                # ĐỒ HỌA CHO TÍNH NĂNG "KHÁM PHÁ D-H"
                elif prac_type == "🔍 Khám phá D-H":
                    T_base = np.eye(4)
                    
                    if exp_mode == "🎬 1. Phân tích D-H (Trực quan hóa)":
                        curr = st.session_state.exp_step
                        # Vẽ những khớp đã hoàn thành
                        for i in range(curr):
                            row = current_dh_df.iloc[i]
                            T_next = np.dot(T_base, dh_transform_matrix(row['θ'], row['d'], row['a'], row['α']))
                            fig.add_trace(go.Scatter3d(x=[T_base[0,3], T_next[0,3]], y=[T_base[1,3], T_next[1,3]], z=[T_base[2,3], T_next[2,3]], mode='lines', line=dict(color=link_colors[i % len(link_colors)], width=18), showlegend=False))
                            T_base = T_next
                            add_cylinder(fig, T_base); draw_axes_3d(fig, T_base, scale=35)
                        
                        if curr < st.session_state.num_joints:
                            gt_row = current_dh_df.iloc[curr]
                            T_target = np.dot(T_base, dh_transform_matrix(gt_row['θ'], gt_row['d'], gt_row['a'], gt_row['α']))
                            add_cylinder(fig, T_target, color='rgba(200,200,200,0.3)', opacity=0.3)
                            draw_axes_3d(fig, T_target, scale=35, opacity=0.3)
                            
                            T1 = np.dot(T_base, trans_z(val_d))
                            T2 = np.dot(T1, rot_z(val_theta))
                            T3 = np.dot(T2, trans_x(val_a))
                            T4 = np.dot(T3, rot_x(val_alpha))
                            fig.add_trace(go.Scatter3d(x=[T_base[0,3], T1[0,3]], y=[T_base[1,3], T1[1,3]], z=[T_base[2,3], T1[2,3]], mode='lines', line=dict(color='yellow', width=5, dash='dot'), showlegend=False))
                            fig.add_trace(go.Scatter3d(x=[T2[0,3], T3[0,3]], y=[T2[1,3], T3[1,3]], z=[T2[2,3], T3[2,3]], mode='lines', line=dict(color='cyan', width=5, dash='dot'), showlegend=False))
                            draw_axes_3d(fig, T4, scale=50); add_cylinder(fig, T4, color='#e74c3c')
                            end_x, end_y, end_z = T4[0,3], T4[1,3], T4[2,3]
                        else:
                            end_x, end_y, end_z = T_base[0,3], T_base[1,3], T_base[2,3]

                    elif exp_mode == "🎲 2. Thử thách giải đố (Blind Test)":
                        add_cylinder(fig, T_base, color='#666666'); draw_axes_3d(fig, T_base, scale=35)
                        target = st.session_state.exp_rand_target
                        if target is not None:
                            T_tar = dh_transform_matrix(target['theta'], target['d'], target['a'], target['alpha'])
                            add_cylinder(fig, T_tar, color='rgba(200,200,200,0.3)', opacity=0.3)
                            draw_axes_3d(fig, T_tar, scale=35, opacity=0.3)
                            
                            T1 = np.dot(T_base, trans_z(r_d)); T2 = np.dot(T1, rot_z(r_th)); T3 = np.dot(T2, trans_x(r_a)); T4 = np.dot(T3, rot_x(r_al))
                            fig.add_trace(go.Scatter3d(x=[T_base[0,3], T1[0,3]], y=[T_base[1,3], T1[1,3]], z=[T_base[2,3], T1[2,3]], mode='lines', line=dict(color='yellow', width=5, dash='dot'), showlegend=False))
                            fig.add_trace(go.Scatter3d(x=[T2[0,3], T3[0,3]], y=[T2[1,3], T3[1,3]], z=[T2[2,3], T3[2,3]], mode='lines', line=dict(color='cyan', width=5, dash='dot'), showlegend=False))
                            draw_axes_3d(fig, T4, scale=50); add_cylinder(fig, T4, color='#e74c3c')
                            end_x, end_y, end_z = T4[0,3], T4[1,3], T4[2,3]
                        else: end_x, end_y, end_z = 0, 0, 0

        fig.update_layout(
            uirevision="khoa_2D_ben_ngoai",
            scene=dict(
                uirevision="khoa_cung_3D_ben_trong",
                xaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                yaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                zaxis=dict(range=[-400, 400], showbackground=False, showticklabels=False), 
                aspectmode='cube'
            ), 
            margin=dict(l=0, r=0, t=0, b=0), height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="robot_3d_plot", theme=None)

    with col_out:
        if mode in ["IK", "FK"]:
            st.markdown("### 🎯 End-Effector")
            st.metric("X", f"{end_x:.1f}"); st.metric("Y", f"{end_y:.1f}"); st.metric("Z", f"{end_z:.1f}")
            if mode == "IK": st.metric("Sai số", f"{np.sqrt((end_x - t_x)**2 + (end_y - t_y)**2 + (end_z - t_z)**2):.1f}")
            st.write("---")
            st.write("**Joint Output**")
            st.write(f"Base Pan: &nbsp;&nbsp;&nbsp; **{th_base:.1f}°**")
            for i, ang in enumerate(angles): st.write(f"theta {i+1}: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{ang:.1f}°**")
                
        elif mode == "DH":
            st.markdown("### 🎯 End-Effector")
            if not practice_mode or (practice_mode and prac_type == "📚 FK: Từng bước" and st.session_state.dh_step_unlocked == st.session_state.num_joints) or (practice_mode and prac_type == "🎯 IK: Giải tích" and st.session_state.ik_calculated_thetas is not None) or (practice_mode and prac_type == "🔍 Khám phá D-H" and exp_mode == "🎬 1. Phân tích D-H (Trực quan hóa)" and st.session_state.exp_step == st.session_state.num_joints):
                st.metric("X", f"{end_x:.1f}"); st.metric("Y", f"{end_y:.1f}"); st.metric("Z", f"{end_z:.1f}")
                st.write("---")
                st.write("**Joint Angles**")
                if not practice_mode:
                    for i in range(st.session_state.num_joints): st.write(f"theta {i+1}: &nbsp;&nbsp;&nbsp;&nbsp; **{dh_angles[i]:.1f}°**")
                elif prac_type == "🎯 IK: Giải tích":
                    for i in range(st.session_state.num_joints): st.write(f"theta {i+1}: &nbsp;&nbsp;&nbsp;&nbsp; **{st.session_state.ik_calculated_thetas[i]:.1f}°**")
            else:
                st.info("Hoàn thành bài tập để xem kết quả")


# =====================================================================
# MODULE 2, 3, 4: CARTESIAN, SCARA, DELTA
# =====================================================================
elif robot_type == "Robot Tuyến tính (Cartesian)":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1: st.title("Cartesian (Gantry) Simulator")

    c_ctrl, c_plot, c_out = st.columns([1, 2.5, 1])
    with c_ctrl:
        st.subheader("Trục Tịnh Tiến")
        px = st.slider("Trục X (Cụm gắp chạy ngang trên cầu)", -200.0, 200.0, 100.0, step=5.0)
        py = st.slider("Trục Y (Cả thanh cầu chạy tới/lui)", -200.0, 200.0, 80.0, step=5.0)
        pz = st.slider("Trục Z (Trục nâng hạ cụm gắp)", -200.0, 0.0, -150.0, step=5.0)
        st.write("---")
        st.info("💡 Chuẩn CNC: Trục Y là 2 thanh ray tĩnh định hướng. Cầu trục vắt ngang là trục X.")
    
    with c_plot:
        fig_cart = go.Figure()
        fig_cart.add_trace(go.Scatter3d(x=[-200, -200], y=[-200, 200], z=[0, 0], mode='lines', line=dict(color='#555555', width=6), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[200, 200], y=[-200, 200], z=[0, 0], mode='lines', line=dict(color='#555555', width=6), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[-200, 200], y=[py, py], z=[0, 0], mode='lines', line=dict(color='#FFD700', width=12), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[px, px], y=[py, py], z=[0, pz], mode='lines', line=dict(color='#FF00FF', width=12), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[-200, 200], y=[py, py], z=[0, 0], mode='markers', marker=dict(size=8, color='white'), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[px], y=[py], z=[0], mode='markers', marker=dict(size=14, color='cyan', line=dict(width=2, color='black')), showlegend=False))
        fig_cart.add_trace(go.Scatter3d(x=[px], y=[py], z=[pz], mode='markers', marker=dict(size=22, color='#FF3366', line=dict(width=3, color='white')), showlegend=False))
        fig_cart.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=-2, z=1)), xaxis_title="X", yaxis_title="Y", zaxis_title="Z", xaxis=dict(range=[-250, 250], showbackground=False), yaxis=dict(range=[-250, 250], showbackground=False), zaxis=dict(range=[-250, 50], showbackground=False), aspectmode='cube'), margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_cart, use_container_width=True)
        
    with c_out:
        st.markdown("### 🎯 End-Effector")
        st.metric("X", f"{px:.1f} mm"); st.metric("Y", f"{py:.1f} mm"); st.metric("Z", f"{pz:.1f} mm")
        st.write("---")
        st.markdown("### ⚙️ Joint Values")
        st.write(f"**d1 (Trục X):** &nbsp;&nbsp; {px:.1f} mm")
        st.write(f"**d2 (Trục Y):** &nbsp;&nbsp; {py:.1f} mm")
        st.write(f"**d3 (Trục Z):** &nbsp;&nbsp; {pz:.1f} mm")

elif robot_type == "SCARA Robot":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1: st.title("SCARA Simulator")

    sc_ctrl, sc_plot, sc_out = st.columns([1, 2.5, 1])
    with sc_ctrl:
        st.subheader("Cấu hình RRP")
        L1 = st.number_input("Chiều dài Link 1 (L1)", 10.0, 300.0, 150.0)
        L2 = st.number_input("Chiều dài Link 2 (L2)", 10.0, 300.0, 120.0)
        st.write("---")
        th1 = st.slider("Khớp xoay 1 (Base)", -150, 150, 30)
        th2 = st.slider("Khớp xoay 2 (Elbow)", -150, 150, -45)
        d3 = st.slider("Khớp tịnh tiến 3 (Trục Z)", -100, 0, -50)
        st.info("💡 SCARA hoạt động theo cơ chế RRP (Xoay - Xoay - Tịnh tiến).")

    with sc_plot:
        fig_sc = go.Figure()
        th1_r, th2_r = np.radians(th1), np.radians(th2)
        x0, y0, z0 = 0, 0, 50
        x1, y1, z1 = L1 * np.cos(th1_r), L1 * np.sin(th1_r), 50
        x2, y2, z2 = x1 + L2 * np.cos(th1_r + th2_r), y1 + L2 * np.sin(th1_r + th2_r), 50
        x3, y3, z3 = x2, y2, 50 + d3
        pts_x, pts_y, pts_z = [x0, x1, x2, x3], [y0, y1, y2, y3], [z0, z1, z2, z3]
        
        fig_sc.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,50], mode='lines', line=dict(color='gray', width=20), showlegend=False))
        colors = ['#FF5733', '#33FF57', '#3357FF']
        for i in range(3):
            fig_sc.add_trace(go.Scatter3d(x=[pts_x[i], pts_x[i+1]], y=[pts_y[i], pts_y[i+1]], z=[pts_z[i], pts_z[i+1]], mode='lines', line=dict(color=colors[i], width=18), showlegend=False))
            fig_sc.add_trace(go.Scatter3d(x=[pts_x[i]], y=[pts_y[i]], z=[pts_z[i]], mode='markers', marker=dict(size=15, color='white', line=dict(width=2, color='black')), showlegend=False))
        fig_sc.add_trace(go.Scatter3d(x=[x3], y=[y3], z=[z3], mode='markers', marker=dict(size=12, color='black', symbol='diamond'), showlegend=False))
        fig_sc.update_layout(scene=dict(xaxis=dict(range=[-300, 300], showbackground=False, showticklabels=False), yaxis=dict(range=[-300, 300], showbackground=False, showticklabels=False), zaxis=dict(range=[-50, 150], showbackground=False, showticklabels=False), aspectmode='cube'), margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_sc, use_container_width=True)

    with sc_out:
        st.markdown("### 🎯 End-Effector")
        st.metric("X", f"{x3:.1f} mm"); st.metric("Y", f"{y3:.1f} mm"); st.metric("Z", f"{z3:.1f} mm")
        st.write("---")
        st.write(f"**θ1:** {th1:.1f}° \n\n **θ2:** {th2:.1f}° \n\n **d3:** {d3:.1f} mm")

elif robot_type == "Delta Robot":
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1: st.title("Delta Parallel Simulator")

    dl_ctrl, dl_plot, dl_out = st.columns([1, 2.5, 1])
    with dl_ctrl:
        st.subheader("Inverse Kinematics (IK)")
        t_x = st.slider("X Target", -100.0, 100.0, 0.0)
        t_y = st.slider("Y Target", -100.0, 100.0, 0.0)
        t_z = st.slider("Z Target", -250.0, -100.0, -180.0)
        st.write("---")
        R_B = st.number_input("Bán kính đế tĩnh (R)", 50, 150, 80)
        R_P = st.number_input("Bán kính mâm gắp (r)", 10, 80, 30)
        L = st.number_input("Tay đòn trên (L)", 50, 200, 100)
        l_arm = st.number_input("Tay đòn dưới (l)", 100, 300, 180)

    def delta_ik(X, Y, Z, R_B, R_P, L, l_arm):
        thetas, pts_u, pts_l = [], [], []
        for alpha_deg in [0, 120, 240]:
            alpha = np.radians(alpha_deg)
            Xt, Yt = X * np.cos(alpha) + Y * np.sin(alpha), -X * np.sin(alpha) + Y * np.cos(alpha)
            Xp, Yp, Zp = Xt + R_P, Yt, Z
            l_eff_sq = l_arm**2 - Yp**2
            if l_eff_sq < 0: return None, None, None 
            l_eff = np.sqrt(l_eff_sq)
            dx, dz = Xp - R_B, Zp - 0
            val = (L**2 + dx**2 + dz**2 - l_eff**2) / (2 * L * np.sqrt(dx**2 + dz**2))
            if abs(val) > 1: return None, None, None 
            theta = np.arctan2(dz, dx) - np.arccos(val)
            thetas.append(np.degrees(theta))
            Xu, Zu = R_B + L * np.cos(theta), L * np.sin(theta)
            pts_u.append((Xu * np.cos(alpha), Xu * np.sin(alpha), Zu))
            pts_l.append((X + R_P * np.cos(alpha), Y + R_P * np.sin(alpha), Z))
        return thetas, pts_u, pts_l

    thetas, pts_u, pts_l = delta_ik(t_x, t_y, t_z, R_B, R_P, L, l_arm)

    with dl_plot:
        fig_dl = go.Figure()
        if thetas is None: st.error("Tọa độ nằm ngoài không gian làm việc của Robot Delta!")
        else:
            bx, by, bz = zip(*[(R_B*np.cos(np.radians(a)), R_B*np.sin(np.radians(a)), 0) for a in [0, 120, 240, 0]])
            fig_dl.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines+markers', line=dict(color='gray', width=8), marker=dict(size=10), showlegend=False))
            px, py, pz = zip(*(pts_l + [pts_l[0]]))
            fig_dl.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers', line=dict(color='orange', width=8), marker=dict(size=6, color='black'), showlegend=False))
            arm_colors = ['#FF3366', '#33FF57', '#3357FF']
            for i in range(3):
                fig_dl.add_trace(go.Scatter3d(x=[bx[i], pts_u[i][0]], y=[by[i], pts_u[i][1]], z=[bz[i], pts_u[i][2]], mode='lines+markers', line=dict(color=arm_colors[i], width=12), marker=dict(size=8, color='white'), showlegend=False))
                fig_dl.add_trace(go.Scatter3d(x=[pts_u[i][0], pts_l[i][0]], y=[pts_u[i][1], pts_l[i][1]], z=[pts_u[i][2], pts_l[i][2]], mode='lines', line=dict(color='cyan', width=5), showlegend=False))
        fig_dl.add_trace(go.Scatter3d(x=[t_x, t_x], y=[t_y, t_y], z=[t_z, t_z - 20], mode='lines', line=dict(color='white', width=10), showlegend=False))
        fig_dl.update_layout(scene=dict(xaxis=dict(range=[-200, 200], showbackground=False, showticklabels=False), yaxis=dict(range=[-200, 200], showbackground=False, showticklabels=False), zaxis=dict(range=[-250, 50], showbackground=False, showticklabels=False), aspectmode='cube'), margin=dict(l=0, r=0, t=0, b=0), height=600)
        st.plotly_chart(fig_dl, use_container_width=True)

    with dl_out:
        if thetas is not None:
            st.markdown("### 🎯 Target")
            st.metric("X", f"{t_x:.1f} mm"); st.metric("Y", f"{t_y:.1f} mm"); st.metric("Z", f"{t_z:.1f} mm")
            st.write("---")
            st.write(f"**M1:** {thetas[0]:.1f}° \n\n **M2:** {thetas[1]:.1f}° \n\n **M3:** {thetas[2]:.1f}°")