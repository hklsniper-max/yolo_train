
import time
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from ultralytics import YOLO


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="실시간 근접 객체 경고",
    page_icon="🚨",
    layout="wide",
)

MODEL_SIZE = 320

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DETECTION_INTERVAL = 2
DEPTH_INTERVAL = 5

DEFAULT_MAX_OBJECT_DISTANCE_M = 2.0
VOICE_WARNING_REPEAT_SEC = 1.5


# ==========================================
# 2. 모델 로딩
# ==========================================

@st.cache_resource
def load_models():

    model_dir = Path(__file__).resolve().parent

    detection_path = model_dir / "yolo11n.pt"
    depth_path = model_dir / "yolo26n-depth.pt"

    missing_models = [
        path.name
        for path in (detection_path, depth_path)
        if not path.is_file()
    ]

    if missing_models:
        raise FileNotFoundError(
            "모델 파일을 찾을 수 없습니다: "
            + ", ".join(missing_models)
        )

    detection_model = YOLO(str(detection_path))
    depth_model = YOLO(str(depth_path))

    inference_lock = threading.Lock()

    return detection_model, depth_model, inference_lock


# ==========================================
# 3. 실시간 영상 처리
# ==========================================

class DistanceWarningProcessor(VideoProcessorBase):

    def __init__(
        self,
        model,
        depth_model,
        inference_lock,
        max_distance_m,
    ):

        self.model = model
        self.depth_model = depth_model
        self.inference_lock = inference_lock

        self.max_distance_m = max_distance_m

        self.frame_count = 0

        self.cached_detections = []
        self.cached_depth_map = None

        self.status_lock = threading.Lock()
        self.warning = ""

        self.previous_time = None
        self.fps = 0.0

    def get_warning(self):

        with self.status_lock:
            return self.warning

    def resize_camera(self, image):

        height, width = image.shape[:2]

        scale = min(
            CAMERA_WIDTH / width,
            CAMERA_HEIGHT / height,
            1.0,
        )

        if scale < 1.0:

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        return image

    def run_detection(self, image):

        result = self.model(
            image,
            imgsz=MODEL_SIZE,
            conf=0.5,
            verbose=False,
            device="cpu",
        )[0]

        detections = []

        boxes = result.boxes

        if boxes is not None:

            for box in boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0].detach().cpu().numpy().astype(int)
                )

                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                detections.append(
                    (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        class_id,
                        confidence,
                    )
                )

        return detections

    def run_depth(self, image):

        result = self.depth_model(
            image,
            imgsz=MODEL_SIZE,
            verbose=False,
            device="cpu",
        )[0]

        if result.depth is None:
            return None

        depth_map = (
            result.depth.data
            .detach()
            .cpu()
            .numpy()
        )

        depth_map = np.squeeze(depth_map)

        if depth_map.ndim != 2:
            return None

        height, width = image.shape[:2]

        if depth_map.shape != (height, width):

            depth_map = cv2.resize(
                depth_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        return depth_map

    def get_object_distance(self, depth_map, box):

        x1, y1, x2, y2, _, _ = box

        height, width = depth_map.shape

        roi_x1 = max(
            0,
            int(x1 + (x2 - x1) * 0.35),
        )

        roi_x2 = min(
            width,
            int(x2 - (x2 - x1) * 0.35),
        )

        roi_y1 = max(
            0,
            int(y1 + (y2 - y1) * 0.35),
        )

        roi_y2 = min(
            height,
            int(y2 - (y2 - y1) * 0.35),
        )

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None

        roi_depth = depth_map[
            roi_y1:roi_y2,
            roi_x1:roi_x2,
        ]

        valid_depth = roi_depth[
            np.isfinite(roi_depth) & (roi_depth > 0)
        ]

        if valid_depth.size == 0:
            return None

        return float(np.median(valid_depth))

    # ======================================
    # 4. 프레임 처리
    # ======================================

    def recv(self, frame: av.VideoFrame):

        image = frame.to_ndarray(format="bgr24")

        image = self.resize_camera(image)

        self.frame_count += 1

        # 객체 탐지: 2프레임마다 실행
        # 첫 프레임에서는 반드시 실행

        run_detection = (
            self.frame_count == 1
            or self.frame_count % DETECTION_INTERVAL == 0
        )

        # 깊이 추정: 5프레임마다 실행

        run_depth = (
            self.cached_depth_map is None
            or self.frame_count % DEPTH_INTERVAL == 0
        )

        # 모델 추론

        with self.inference_lock:

            if run_detection:

                self.cached_detections = self.run_detection(
                    image
                )

            if run_depth:

                self.cached_depth_map = self.run_depth(
                    image
                )

        depth_map = self.cached_depth_map

        nearby_objects = []

        warning = ""

        # ==================================
        # 5. 객체 거리 계산
        # ==================================

        if depth_map is not None:

            for box in self.cached_detections:

                x1, y1, x2, y2, class_id, confidence = box

                distance_m = self.get_object_distance(
                    depth_map,
                    box,
                )

                if distance_m is None:
                    continue

                if distance_m <= self.max_distance_m:

                    area = (x2 - x1) * (y2 - y1)

                    nearby_objects.append(
                        (
                            area,
                            box,
                            distance_m,
                        )
                    )

        # ==================================
        # 6. 가장 큰 근접 객체 선택
        # ==================================

        if nearby_objects:

            _, largest_box, distance_m = max(
                nearby_objects,
                key=lambda item: item[0],
            )

            x1, y1, x2, y2, class_id, confidence = largest_box

            label = self.model.names[class_id]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                image,
                f"{label} {confidence:.0%} {distance_m:.1f}m",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

            warning = (
                f"경고! {distance_m:.1f}미터 앞에 "
                f"{label}이 있습니다."
            )

        else:

            cv2.putText(
                image,
                f"No object within {self.max_distance_m:.1f}m",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

        # ==================================
        # 7. 경고 상태 저장
        # ==================================

        with self.status_lock:
            self.warning = warning

        # ==================================
        # 8. FPS 계산
        # ==================================

        current_time = time.perf_counter()

        if self.previous_time is not None:

            elapsed = current_time - self.previous_time

            if elapsed > 0:

                instant_fps = 1.0 / elapsed

                if self.fps == 0:
                    self.fps = instant_fps
                else:
                    self.fps = (
                        self.fps * 0.8
                        + instant_fps * 0.2
                    )

        self.previous_time = current_time

        cv2.putText(
            image,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"Frame: {self.frame_count}",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ==========================================
# 9. 브라우저 음성 경고
# ==========================================

_SPEECH_ALERT = st.components.v2.component(
    "distance_warning_speech",

    html="""
    <span class="speech-status" aria-live="polite"></span>
    """,

    js="""
    const instances = new WeakMap();

    export default function ({ data, parentElement }) {

        const status = parentElement.querySelector(
            ".speech-status"
        );

        let state = instances.get(parentElement);

        if (!state) {

            state = {
                text: "",
                timer: null,
                speaking: false
            };

            instances.set(parentElement, state);
        }

        if (!window.speechSynthesis) {

            status.textContent =
                "이 브라우저는 음성 출력을 지원하지 않습니다.";

            return;
        }

        status.textContent = "";

        const nextText = data?.text ?? "";

        const repeatDelayMs = data?.repeatDelayMs ?? 1500;

        if (nextText !== state.text) {

            if (state.timer !== null) {

                window.clearTimeout(state.timer);
                state.timer = null;
            }

            window.speechSynthesis.cancel();

            state.text = nextText;
            state.speaking = false;
        }

        const speak = () => {

            if (!parentElement.isConnected || !state.text) {
                return;
            }

            if (state.speaking) {
                return;
            }

            state.speaking = true;

            const utterance = new SpeechSynthesisUtterance(
                state.text
            );

            utterance.lang = "ko-KR";
            utterance.rate = 1.1;
            utterance.volume = 1.0;

            utterance.onend = () => {

                state.speaking = false;

                if (state.text) {

                    state.timer = window.setTimeout(
                        () => {

                            state.timer = null;
                            speak();

                        },
                        repeatDelayMs
                    );
                }
            };

            utterance.onerror = () => {
                state.speaking = false;
            };

            window.speechSynthesis.speak(utterance);
        };

        if (
            state.text &&
            state.timer === null &&
            !state.speaking &&
            !window.speechSynthesis.speaking
        ) {
            speak();
        }
    }
    """,
)


# ==========================================
# 10. Streamlit 화면
# ==========================================

st.title("🚨 실시간 근접 객체 경고")

st.write(
    "웹캠 영상을 분석하여 객체를 탐지하고 "
    "가까운 객체에 대해 음성으로 경고합니다."
)

st.caption(
    "YOLO11 객체 탐지 + YOLO26 깊이 추정"
)

st.divider()

max_distance_m = st.slider(
    "경고 거리 (미터)",
    min_value=0.5,
    max_value=5.0,
    value=DEFAULT_MAX_OBJECT_DISTANCE_M,
    step=0.1,
)

camera_enabled = st.toggle(
    "카메라 분석 활성화",
    value=False,
)

if not camera_enabled:

    st.info(
        "카메라 분석을 활성화한 다음 "
        "브라우저에서 카메라 권한을 허용하세요."
    )

    st.stop()


# ==========================================
# 11. 모델 불러오기
# ==========================================

try:

    with st.spinner("AI 모델을 불러오는 중입니다..."):

        model, depth_model, inference_lock = load_models()

except Exception as exc:

    st.error("AI 모델 로딩에 실패했습니다.")

    st.exception(exc)

    st.stop()


# ==========================================
# 12. 웹캠 연결
# ==========================================

webrtc_ctx = webrtc_streamer(

    key="distance-warning-camera",

    mode=WebRtcMode.SENDRECV,

    video_processor_factory=lambda: DistanceWarningProcessor(
        model,
        depth_model,
        inference_lock,
        max_distance_m,
    ),

    media_stream_constraints={
        "video": {
            "width": {"ideal": CAMERA_WIDTH},
            "height": {"ideal": CAMERA_HEIGHT},
            "frameRate": {"ideal": 15, "max": 20},
        },
        "audio": False,
    },

    async_processing=True,
)


# ==========================================
# 13. 경고 화면 갱신
# ==========================================

if webrtc_ctx.video_processor is None:

    st.info(
        "위의 START 버튼을 눌러 카메라 분석을 시작하세요."
    )

else:

    @st.fragment(run_every="1s")
    def show_warning(processor):

        warning = processor.get_warning()

        if warning:

            st.error(warning)

        else:

            st.success(
                f"{max_distance_m:.1f}m 이내에 "
                "감지된 객체가 없습니다."
            )

        _SPEECH_ALERT(
            data={
                "text": warning,
                "repeatDelayMs": int(
                    VOICE_WARNING_REPEAT_SEC * 1000
                ),
            },
            key="distance-warning-speech",
        )

    show_warning(webrtc_ctx.video_processor)
#a