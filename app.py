# -*- coding: utf-8 -*-
import cv2
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import torch
from mtcnn import MTCNN
from openai import OpenAI
from PIL import Image, ImageFile, ImageDraw
from io import BytesIO
import logging
from torchvision.transforms import transforms
from skimage import feature

# 图像处理配置
ImageFile.LOAD_TRUNCATED_IMAGES = True
MEDIAN_FILTER_SIZE = 5

# 初始化日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 页面配置
st.set_page_config(
    page_title="健康先知 - 专业健康分析",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def apply_custom_style():
    st.markdown(f"""
    <style>
        /* 保持原有样式不变 */
        .stTextInput label, 
        .stNumberInput label,
        .stTextArea label,
        .stSelectbox label,
        .stFileUploader label,
        .stCheckbox label,
        .stRadio label {{
            color: #000000 !important;
        }}
        body, .stApp, .stAlert, .stMarkdown {{
            color: #000000 !important;
        }}
        .stAlert svg,
        .stSuccess svg,
        .stWarning svg {{
            fill: #000000 !important;
        }}
        .stApp {{
            background-color: #808080;
            color: #000000 !important;
        }}
        
        .stTextInput input, 
        .stNumberInput input,
        .stTextArea textarea {{
            background-color: #ffffe0 !important;
            color: #000000 !important;
            border-radius: 8px;
        }}
        .stSelectbox select {{
            background-color: #ffffff !important;
            color: #000000 !important;
        }}
        .stFileUploader > div {{
            background-color: #ffffff !important;
            border: 2px dashed #4CAF50;
        }}
        /* 新增表单提交按钮专属样式 */
        div[data-testid="stFormSubmitButton"] button {{
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF !important;
        }}
        div[data-testid="stFormSubmitButton"] button:hover {{
            color: #FF0000 !important;
            background-color: #333333 !important;
            border-color: #FF0000 !important;
        }}
        .stButton button {{
            background-color: #000000 !important;  /* 修改背景色为黑色 */
            color: #FFFFFF !important;            /* 修改文字颜色为白色 */
            border-radius: 8px;
            padding: 0.5rem 2rem;
            border: 1px solid #ffffff;           /* 添加白色边框 */
            transition: all 0.3s ease;           /* 添加过渡动画 */
        }}
        /* 悬停状态样式 */
        .stButton button:hover {{
            color: #FF0000 !important;           /* 悬停时文字变红 */
            background-color: #333333 !important; /* 悬停时背景色稍亮 */
            border-color: #FF0000;               /* 悬停时边框变红 */
        }}
        .stAlert {{ 
            background-color: #ffebee !important;
        }}
        .stDataFrame {{
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .medical-image {{
            border: 2px solid #2196F3;
            border-radius: 8px;
            padding: 5px;
        }}
        .face-marker {{
            border: 2px solid #FF5722 !important;
            border-radius: 50%;
            padding: 2px;
        }}
    </style>
    """, unsafe_allow_html=True)


apply_custom_style()


@st.cache_resource
def load_medical_models():
    face_detector = MTCNN(
        min_face_size=80,
        steps_threshold=[0.7, 0.8, 0.9],
        scale_factor=0.8
    )
    tongue_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    tongue_model.fc = torch.nn.Linear(512, 4)
    return face_detector, tongue_model


def get_deepseek_client():
    try:
        return OpenAI(
            api_key=st.secrets["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1"
        )
    except KeyError:
        st.error("未找到API密钥，请检查secrets.toml配置")
        st.stop()
    except Exception as e:
        st.error(f"客户端初始化失败: {str(e)}")
        st.stop()


class HealthScoreProcessor:
    @staticmethod
    def parse_scores(text):
        """改进的评分解析方法"""
        score_patterns = {
            r'症状痊愈能力[^\d:]*评分': '症状痊愈能力评分',
            r'饮食[^\d:]*评分': '饮食评分',
            r'代谢[^\d:]*评分': '代谢评分',
            r'睡眠[^\d:]*评分': '睡眠评分'
        }

        scores = {}
        for pattern, name in score_patterns.items():
            match = re.search(fr'{pattern}:\s*(\d+)/10', text)
            if match:
                scores[name] = int(match.group(1))
            else:
                scores[name] = 5  # 默认值
        return scores


class ChartGenerator:
    @staticmethod
    def create_radar_chart(scores):
        # 保持原有雷达图实现不变
        categories = list(scores.keys())
        values = list(scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line=dict(color='#4CAF50'),
            name='健康指标'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 10], gridcolor='#cccccc'),
                angularaxis=dict(gridcolor='#cccccc')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin = dict(l=20, r=20, t=40, b=20)
        )
        return fig


def medical_preprocess(image_bytes):
    try:
        buffer = BytesIO(image_bytes)
        img = Image.open(buffer).convert('RGB')
        img_array = np.array(img)
        filtered = cv2.medianBlur(img_array, MEDIAN_FILTER_SIZE)
        return Image.fromarray(filtered)
    except Exception as e:
        logging.error(f"图像预处理失败: {str(e)}")
        return None


class MedicalAnalyzer:
    def __init__(self):
        self.face_detector, self.tongue_model = load_medical_models()
        self.tongue_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.eye_roi_scale = 1.5

    def _get_eye_regions(self, img_array, keypoints):
        """修正眼部ROI获取逻辑"""
        eye_regions = []
        try:
            for eye_type in ['left_eye', 'right_eye']:
                if eye_type not in keypoints:
                    continue

                # 获取双眼坐标计算距离
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']
                eye_distance = np.sqrt((left_eye[0] - right_eye[0]) ** 2 + (left_eye[1] - right_eye[1]) ** 2)

                # 计算ROI参数
                x_center, y_center = keypoints[eye_type]
                base_size = eye_distance * 0.3
                width = int(base_size * self.eye_roi_scale)
                height = int(width * 0.6)

                # 计算坐标边界
                x1 = max(0, x_center - width // 2)
                y1 = max(0, y_center - height // 2)
                x2 = min(img_array.shape[1], x_center + width // 2)
                y2 = min(img_array.shape[0], y_center + height // 2)

                roi = img_array[y1:y2, x1:x2]
                if roi.size > 0:
                    eye_regions.append(roi)
        except Exception as e:
            logging.error(f"眼部区域获取失败: {str(e)}")
        return eye_regions

    def _analyze_dark_circles(self, eye_roi):
        try:
            lab = cv2.cvtColor(eye_roi, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # 颜色分析
            a_mean = np.mean(a_channel)
            l_std = np.std(l_channel)

            # 纹理分析
            radius = 2
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(l_channel, n_points, radius, method="uniform")
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3))
            hist = hist.astype("float") / hist.sum()
            entropy = -np.sum(hist * np.log(hist + 1e-10))

            severity = 0.4 * a_mean + 0.3 * l_std + 0.3 * entropy
            return np.clip(severity / 100, 0, 1)
        except Exception as e:
            logging.error(f"黑眼圈分析失败: {str(e)}")
            return 0.0

    def detect_dark_circles(self, image: Image.Image, keypoints: dict) -> float:
        try:
            img_array = np.array(image)
            eye_rois = self._get_eye_regions(img_array, keypoints)
            if not eye_rois:
                return 0.0

            scores = []
            for roi in eye_rois:
                if roi.size == 0:
                    continue
                score = self._analyze_dark_circles(roi)
                scores.append(score)

            return round(np.mean(scores), 2) if scores else 0.0
        except Exception as e:
            logging.error(f"黑眼圈检测失败: {str(e)}")
            return 0.0

    def analyze_face(self, image: Image.Image) -> dict:
        try:
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

            results = self.face_detector.detect_faces(img_array)
            if not results:
                return None

            main_face = max(results, key=lambda x: x['confidence'])
            keypoints = main_face['keypoints']

            # 黑眼圈检测
            dark_circle_score = self.detect_dark_circles(image, keypoints)

            # 皮肤分析
            ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
            _, cr, _ = cv2.split(ycrcb)
            skin_score = np.clip(cr.mean() / 140, 0, 1)

            return {
                'dark_circle': dark_circle_score,
                'skin_score': round(skin_score, 4),
                'keypoints': keypoints
            }
        except Exception as e:
            logging.error(f"面部分析错误: {str(e)}")
            return None

    def analyze_tongue(self, image):
        try:
            img_tensor = self.tongue_transform(image).unsqueeze(0)
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
                self.tongue_model.cuda()

            with torch.no_grad():
                outputs = self.tongue_model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)

            return {
                'diagnosis': ['正常', '湿热', '血瘀', '气虚'][torch.argmax(probs).item()],
                'confidence': probs[0].tolist(),
                'heatmap': self._generate_heatmap(image)
            }
        except Exception as e:
            logging.error(f"舌诊分析错误: {str(e)}")
            return None

    def _generate_heatmap(self, image):
        img_array = np.array(image)
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        _, cr, _ = cv2.split(ycrcb)
        heatmap = cv2.applyColorMap(cr, cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))


def main_app():
    client = get_deepseek_client()
    analyzer = MedicalAnalyzer()
    st.title("🏥 健康先知智能分析系统demo")
    st.markdown("---")

    with st.form("health_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🆔 基本信息")
            gender = st.selectbox("性别*", ["男", "女"], index=0)
            age = st.number_input("年龄*", min_value=1, max_value=120, value=30, step=1)
            height = st.number_input("身高(cm)*", min_value=50, max_value=250, value=170, step=1)
            weight = st.number_input("体重(kg)*", min_value=20, max_value=300, value=65, step=1)
            sleep = st.selectbox("睡眠情况*", ["良好(睡眠7h以上，无噩梦，入睡快，中途无醒）", "一般（睡眠4-7h，多梦，入睡较慢，中途偶尔醒）", "差（睡眠<4h，难以入睡）"], index=1)

        with col2:
            st.subheader("💊 健康指标")
            symptoms = st.text_area("症状描述（如有慢性病需填入）*", help="例：持续三天头痛，伴有轻微发热...", height=120)
            diet = st.text_input("近期饮食", help="例：最近三天以快餐为主...")
            temp = st.number_input("体温(℃)", min_value=35.0, max_value=42.0, value=36.5, step=0.1)
            heart_rate = st.number_input("心率(bpm)", min_value=30, max_value=200, value=75)

        st.subheader("📸 医学影像上传")
        img_col1, img_col2 = st.columns(2)
        face_img = img_col1.file_uploader("面部照片", type=["jpg", "png", "jpeg"])
        tongue_img = img_col2.file_uploader("舌苔照片", type=["jpg", "png", "jpeg"])

        submitted = st.form_submit_button("🚀 开始全面分析")

    if submitted:
        missing = []
        if not gender: missing.append("性别")
        if not age: missing.append("年龄")
        if not height: missing.append("身高")
        if not weight: missing.append("体重")
        if not symptoms.strip(): missing.append("症状描述")
        if not sleep: missing.append("睡眠情况")

        if missing:
            st.error(f"缺少必填项: {', '.join(missing)}")
            st.stop()

        try:
            face_data, tongue_data = None, None
            processed_face, processed_tongue = None, None

            if face_img:
                processed_face = medical_preprocess(face_img.getvalue())
                if processed_face:
                    face_data = analyzer.analyze_face(processed_face)

            if tongue_img:
                processed_tongue = medical_preprocess(tongue_img.getvalue())
                if processed_tongue:
                    tongue_data = analyzer.analyze_tongue(processed_tongue)

            analysis_data = {
                "人口统计": f"{gender}性 | 年龄 {age}岁 | 身高 {height}cm | 体重 {weight}kg",
                "症状描述": symptoms,
                "饮食情况": diet if diet.strip() else "未提供详细信息",
                "生理指标": f"体温 {temp}℃ | 心率 {heart_rate}bpm" if temp or heart_rate else "无额外生理指标",
                "睡眠质量": sleep,
                "面部特征": str(face_data) if face_data else "未检测到人脸",
                "舌苔诊断": str(tongue_data) if tongue_data else "未提供舌苔图像"
            }

            # 修改后的API提示词
            with st.spinner("🔍 正在分析基础数据..."):
                chat_prompt = f"""作为资深全科医生，请按以下要求分析健康数据：
                {analysis_data}

                必须包含以下四个评分项（名称严格匹配）：
                1. 症状痊愈能力评分：根据症状描述的痊愈能力评分（0-10分）
                2. 饮食评分：根据饮食健康程度评分（0-10分）
                3. 代谢评分：根据BMI（男/女）、面部特征和生理指标评分（0-10分）
                4. 睡眠评分：根据睡眠质量、面部特征评分（0-10分）

                输出格式必须严格遵循：
                ### 症状痊愈能力评分: [数值]/10
                ### 饮食评分: [数值]/10
                ### 代谢评分: [数值]/10
                ### 睡眠评分: [数值]/10
                ### 潜在风险: [内容]
                ### 建议框架: [内容]"""

                chat_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": chat_prompt}],
                    temperature=0.3,
                    max_tokens=1024
                ).choices[0].message.content

            # 解析评分
            processor = HealthScoreProcessor()
            final_scores = processor.parse_scores(chat_response)

            # 验证评分
            required_scores = ['症状痊愈能力评分', '饮食评分', '代谢评分', '睡眠评分']
            if not all(key in final_scores for key in required_scores):
                st.warning("部分评分解析失败，使用默认值")
                final_scores = {key: final_scores.get(key, 5) for key in required_scores}

            with st.spinner("🧠 进行深度推理分析..."):
                reasoner_prompt = f"""基于以下分析结果：
                {chat_response}

                请完成：
                1. 生成综合健康评分（0-100分制）
                2. 创建详细的分项健康评分表（症状、饮食、代谢、睡眠）
                3. 预测或判断可能的症状
                4. 给出分步骤的专业建议
                5. 列出需要警惕的危险信号
                6. 预估症状恢复时间周期并给出依据

                要求：
                • 区分[思维链]和[正式诊断]
                • 评分标准与其他健康指标保持一致
                • 使用中文且易于理解"""

                reasoner_response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": reasoner_prompt}],
                    temperature=0.2,
                    max_tokens=2048
                ).choices[0].message.content

            st.success("✅ 分析完成！")
            st.markdown("---")

            if face_data and processed_face:
                with st.expander("🧑 面部健康报告", expanded=True):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        circle_score = face_data.get('dark_circle', 0)
                        status = "正常" if circle_score < 0.3 else "轻度" if circle_score < 0.6 else "重度"
                        st.metric("🟤 黑眼圈指数",
                                  f"{circle_score * 10:.1f}/10 ({status})",
                                  help="0-3:正常, 4-6:轻度, 7-10:重度")
                        st.metric("皮肤健康度",
                                  f"{face_data['skin_score'] * 10:.1f}/10",
                                  help="基于色度分析，数值越高越好")
                    with cols[1]:
                        draw_img = processed_face.copy()
                        draw = ImageDraw.Draw(draw_img)
                        for name, (x, y) in face_data['keypoints'].items():
                            draw.ellipse([x - 10, y - 10, x + 10, y + 10], outline="#FF5722", width=3)
                        st.image(draw_img,
                                 caption="面部特征点分析",
                                 use_column_width=True)

            if tongue_data and processed_tongue:
                with st.expander("👅 舌象健康报告", expanded=True):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        st.metric("诊断结果", tongue_data['diagnosis'])
                        st.metric("综合置信度", f"{max(tongue_data['confidence']) * 100:.1f}%")
                    with cols[1]:
                        st.image(tongue_data['heatmap'],
                                 caption="舌体热力图分析",
                                 use_column_width=True)

            with st.expander("📄 查看完整诊断报告", expanded=True):
                st.subheader("初步分析结果")
                st.markdown(chat_response)
                st.subheader("综合诊断报告")
                st.markdown(reasoner_response)

            processor = HealthScoreProcessor()
            scores = processor.parse_scores(chat_response)
            final_scores = {
                "症状痊愈能力评分": scores.get("症状痊愈能力评分", 5),
                "饮食评分": scores.get("饮食评分", 5),
                "代谢评分": scores.get("代谢评分", 5),
                "睡眠评分": scores.get("睡眠评分", 5)
            }

            # 健康指标分析部分
            st.subheader("📊 健康指标可视化")
            col_left, col_right = st.columns([1, 2])

            with col_left:
                df = pd.DataFrame.from_dict(final_scores, orient='index', columns=['评分'])
                st.dataframe(
                    df.style.format(precision=0)
                    .highlight_between(left=0, right=5, color='#000000')
                    .highlight_between(left=6, right=8, color='#000000')
                    .highlight_between(left=9, right=10, color='#000000'),
                    use_container_width=True
                )

            with col_right:
                st.plotly_chart(
                    ChartGenerator.create_radar_chart(final_scores),
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"系统发生严重错误: {str(e)}")
            logging.exception("系统崩溃")
            st.stop()


if __name__ == "__main__":
    try:
        main_app()
    except ImportError as e:
        st.error(f"缺少依赖库: {str(e)}\n请执行: pip install mtcnn torchvision opencv-python-headless")
