import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


# === Примеры файлов для скачивания ===
def get_example_csv_bytes():
    data = np.array([
        [-1, -1.73205],
        [2, 0],
        [-1, 1.73205]
    ])
    df = pd.DataFrame(data, columns=["x", "y"])
    return df.to_csv(index=False).encode("utf-8")



# =================== MDM Algorithm ===================

def compute_b(A):
    return 0.5 * np.sum(A**2, axis=0)

def Q(u, A, b):
    return 0.5 * np.dot((A.T @ A @ u), u) - np.dot(b, u)

def mdm_algorithm(A, u0, eps=5e-3, max_iter=1000):
    b = compute_b(A)
    u = u0.copy()
    x = A @ u
    R_hist = []
    delta_hist = []
    x_hist = [x.copy()]
    u_hist = [u.copy()]
    Iplus_hist = []

    for _ in range(max_iter):
        inner = A.T @ x - b
        I_plus = np.where(u > 0)[0]
        i_min = np.argmin(inner)
        i_max = I_plus[np.argmax(inner[I_plus])]
        d = inner[i_max] - inner[i_min]

        delta_hist.append(np.sqrt(d))
        Iplus_hist.append(len(I_plus))
        R = np.sqrt(-2 * Q(u, A, b))
        R_hist.append(R)

        if np.sqrt(d) < eps:
            break

        ai, aj = A[:, i_min], A[:, i_max]
        denom = np.linalg.norm(ai - aj) ** 2
        delta_star = d / denom
        delta_val = min(delta_star, u[i_max])

        u[i_min] += delta_val
        u[i_max] -= delta_val
        x = A @ u

        x_hist.append(x.copy())
        u_hist.append(u.copy())

    return np.array(x_hist), np.array(u_hist), np.array(delta_hist), R_hist, Iplus_hist

# =================== Examples ===================

def get_example(name):
    if name == "Пример 1: 3 точки":
        a1 = np.array([-1, -np.sqrt(3)])
        a2 = np.array([2, 0])
        a3 = np.array([-1, np.sqrt(3)])
        return np.array([a1, a2, a3])
    elif name == "Пример 1: +центр масс":
        a1 = np.array([-1, -np.sqrt(3)])
        a2 = np.array([2, 0])
        a3 = np.array([-1, np.sqrt(3)])
        a4 = np.mean([a1, a2, a3], axis=0)
        return np.array([a1, a2, a3, a4])
    elif name == "Пример 2: 9 точек на окружности":
        np.random.seed(42)
        center = np.array([20, 30])
        radius = 10
        angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
        points = np.array([center + radius*np.array([np.cos(a), np.sin(a)]) for a in angles])
        centroid = np.mean(points, axis=0)
        return np.vstack([points, centroid])
    elif name == "Пример 3: 10 точек в круге":
        np.random.seed(42)
        center = np.array([20, 30])
        radius = 10
        return center + (np.random.rand(10, 2)*2 - 1)*radius
    elif name == "Пример 4: 100 точек в круге":
        np.random.seed(42)
        center = np.array([20, 30])
        radius = 10
        return center + (np.random.rand(100, 2)*2 - 1)*radius

# =================== Plotting ===================

def plot_solution(points, x_list, return_bytes=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(points[:, 0], points[:, 1], 'bo', label='Точки')
    ax.plot(x_list[:, 0], x_list[:, 1], 'k--', alpha=0.6, label='Траектория $x_k$')
    ax.plot(x_list[-1, 0], x_list[-1, 1], 'go', label='Центр круга $x^*$')
    r = np.max(np.linalg.norm(points - x_list[-1], axis=1))
    circle = plt.Circle(x_list[-1], r, color='g', fill=False, label='Минимальный круг')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title("Решение задачи Сильвестра")
    ax.legend()
    ax.grid(True)
    if return_bytes:
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    else:
        st.pyplot(fig)

def plot_convergence(x_list, delta_hist, R_hist, Iplus_hist, return_bytes=False):
    norm_errors = [np.linalg.norm(x - x_list[-1]) for x in x_list]
    ratios = [norm_errors[i+1]/norm_errors[i] if norm_errors[i] != 0 else 0
              for i in range(len(norm_errors)-1)]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(R_hist)
    axs[0, 0].set_title("R_k — радиус минимального круга")

    axs[0, 1].plot(norm_errors, label="||x_k - x*||")
    axs[0, 1].plot(delta_hist, label="√Δ(u_k)")
    axs[0, 1].legend()
    axs[0, 1].set_title("Показатели сходимости")

    axs[1, 0].plot(Iplus_hist)
    axs[1, 0].set_title("|I⁺(u_k)| — мощность носителя")

    axs[1, 1].plot(ratios)
    axs[1, 1].set_title("Отношение ||x_{k+1} - x*|| / ||x_k - x*||")

    plt.tight_layout()
    if return_bytes:
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()
    else:
        st.pyplot(fig)

# =================== Streamlit App ===================

st.title("Модифицированный МДМ-алгоритм")

with st.sidebar:
    mode = st.radio("Выберите источник данных:", ["📁 Загрузить свой файл", "📊 Выбрать пример"])

    if mode == "📁 Загрузить свой файл":
        uploaded = st.file_uploader("Загрузите CSV или TXT файл (2D точки)", type=["csv", "txt"])
        data = None
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded, header=None)
                if df.shape[1] < 2:
                    st.error("Ошибка: файл должен содержать хотя бы 2 числовых столбца (x, y).")
                elif df.shape[0] < 3:
                    st.error("Ошибка: необходимо минимум 3 точки для построения минимального круга.")
                else:
                    df_clean = df.dropna().iloc[:, :2]
                    # if not np.issubdtype(df_clean.dtypes[0], np.number) or not np.issubdtype(df_clean.dtypes[1],
                    #                                                                          np.number):
                    #     st.error("Ошибка: данные должны быть числовыми.")
                    # else:
                    data = get_example("Пример 2: 9 точек на окружности")
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

    else:
        example_name = st.selectbox("Выберите пример", [
            "Пример 1: 3 точки",
            "Пример 1: +центр масс",
            "Пример 2: 9 точек на окружности",
            "Пример 3: 10 точек в круге",
            "Пример 4: 100 точек в круге"
        ])
        data = get_example(example_name)

if data is not None:
    st.subheader("Результат")
    A = data.T
    u0 = np.zeros(data.shape[0])
    u0[-1] = 1.0 if mode == "📊 Выбрать пример" and "центр масс" in example_name else 0.0
    u0[0] = 1.0 if u0.sum() == 0 else u0[0]

    x_list, u_list, delta_hist, R_hist, Iplus_hist = mdm_algorithm(A, u0)

    if st.button("🖼️ Показать результат на графике"):
        plot_solution(data, x_list)

    img_bytes = plot_solution(data, x_list, return_bytes=True)
    st.download_button("📥 Скачать график результата(PNG)", data=img_bytes, file_name="mdm_result.png")

    if st.button("📈 Показать графики сходимости"):
        plot_convergence(x_list, delta_hist, R_hist, Iplus_hist)

    img_bytes_2 = plot_convergence(x_list, delta_hist, R_hist, Iplus_hist, return_bytes=True)
    st.download_button("📥 Скачать графики сходимости(PNG)", data=img_bytes_2, file_name="mdm_result_convergence.png")

# st.markdown("---")
# st.markdown("### 📁 Примеры файлов для скачивания")

# st.download_button(
#     label="📥 Скачать CSV-пример",
#     data=get_example_csv_bytes(),
#     file_name="example_points.csv",
#     mime="text/csv"
# )
