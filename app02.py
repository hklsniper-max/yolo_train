import threading
import time
from pathlib import Path
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


st.set_page_config(page_title="실시간 근접 객체 경고", page_icon="🚨")

MODEL_SIZE = 480
DEFAULT_MAX_OBJECT_DISTANCE_M = 2.0
VOICE_WARNING_REPEAT_SEC = 1.5


@st.cache_resource
def load_models():
    model_dir = Path(__file__).resolve().parent
    detection_path = model_dir / "yolo11n.pt"
    depth_path = model_dir / "yolo26n-depth.pt"

    missing_models = [
        path.name for path in (detection_path, depth_path) if not path.is_file()
    ]
    if missing_models:
        raise FileNotFoundError(
            "모델 파일을 찾을 수 없습니다: " + ", ".join(missing_models)
        )

    return YOLO(str(detection_path)), YOLO(str(depth_path)), threading.Lock()


class DistanceWarningProcessor(VideoProcessorBase):
    def __init__(self, model, depth_model, inference_lock, max_distance_m):
        self.model = model
        self.depth_model = depth_model
        self.inference_lock = inference_lock
        self.max_distance_m = max_distance_m
        self.status_lock = threading.Lock()
        self.warning = ""
        self.previous_time = time.perf_counter()

    def get_warning(self):
        with self.status_lock:
            return self.warning

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        with self.inference_lock:
            detection = self.model(
                image, imgsz=MODEL_SIZE, conf=0.5, verbose=False
            )[0]
            depth_result = self.depth_model(
                image, imgsz=MODEL_SIZE, verbose=False
            )[0]

        if depth_result.depth is None:
            raise RuntimeError("깊이 추정 결과를 가져올 수 없습니다.")

        depth_map = depth_result.depth.data.cpu().numpy()
        height, width = image.shape[:2]
        if depth_map.shape != (height, width):
            depth_map = cv2.resize(
                depth_map,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )

        nearby_objects = []
        boxes = detection.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi_x1 = max(0, int(x1 + (x2 - x1) * 0.35))
                roi_x2 = min(width, int(x2 - (x2 - x1) * 0.35))
                roi_y1 = max(0, int(y1 + (y2 - y1) * 0.35))
                roi_y2 = min(height, int(y2 - (y2 - y1) * 0.35))
                roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_depth = roi_depth[
                    np.isfinite(roi_depth) & (roi_depth > 0)
                ]
                if valid_depth.size == 0:
                    continue

                distance_m = float(np.median(valid_depth))
                if distance_m <= self.max_distance_m:
                    nearby_objects.append(
                        (
                            (x2 - x1) * (y2 - y1),
                            box,
                            (x1, y1, x2, y2),
                            distance_m,
                        )
                    )

        warning = ""
        if nearby_objects:
            _, largest_box, (x1, y1, x2, y2), distance_m = max(
                nearby_objects, key=lambda item: item[0]
            )
            label = self.model.names[int(largest_box.cls[0].item())]
            confidence = float(largest_box.conf[0].item())
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"Largest: {label} {confidence:.0%} {distance_m:.1f}m",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            warning = f"경고! {distance_m:.1f}미터 앞에 {label}이 있습니다."
        else:
            cv2.putText(
                image,
                f"No object within {self.max_distance_m:.1f}m",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        with self.status_lock:
            self.warning = warning

        current_time = time.perf_counter()
        fps = 1 / max(current_time - self.previous_time, 1e-5)
        self.previous_time = current_time
        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


_SPEECH_ALERT = st.components.v2.component(
    "distance_warning_speech",
    html="<span class='speech-status' aria-live='polite'></span>",
    js="""
const instances = new WeakMap();

export default function ({ data, parentElement }) {
  const status = parentElement.querySelector(".speech-status");
  let state = instances.get(parentElement);
  if (!state) {
    state = { text: "", timer: null };
    instances.set(parentElement, state);
  }

  if (!window.speechSynthesis) {
    status.textContent = "이 브라우저는 음성 출력을 지원하지 않습니다.";
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
    window.speechSynthesis?.cancel();
    state.text = nextText;
  }

  const speak = () => {
    if (!parentElement.isConnected || !state.text) return;
    if (!window.speechSynthesis) return;

    const utterance = new SpeechSynthesisUtterance(state.text);
    utterance.lang = "ko-KR";
    utterance.rate = 1.1;
    utterance.onend = () => {
      state.timer = window.setTimeout(() => {
        state.timer = null;
        speak();
      }, repeatDelayMs);
    };
    window.speechSynthesis.speak(utterance);
  };

  if (state.text && state.timer === null && !window.speechSynthesis.speaking) {
    speak();
  }
}
""",
)


st.title("실시간 근접 객체 경고")
st.write(
    "브라우저 카메라 영상에서 객체를 찾고, 가까운 객체의 거리와 음성 경고를 표시합니다."
)

max_distance_m = st.slider(
    "경고 거리 (미터)",
    min_value=0.5,
    max_value=5.0,
    value=DEFAULT_MAX_OBJECT_DISTANCE_M,
    step=0.1,
)

camera_enabled = st.toggle("카메라 분석 활성화")
if not camera_enabled:
    st.info("분석을 켜고 브라우저에서 카메라 사용을 허용하세요.")
    st.caption("웹 배포 시 HTTPS 연결이 필요합니다.")
    st.stop()

try:
    with st.spinner("객체 탐지 및 깊이 추정 모델을 불러오는 중입니다..."):
        model, depth_model, inference_lock = load_models()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

webrtc_ctx = webrtc_streamer(
    key="distance-warning-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: DistanceWarningProcessor(
        model, depth_model, inference_lock, max_distance_m
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor is None:
    st.caption("카메라 시작 버튼을 누르면 실시간 분석이 시작됩니다.")
else:

    @st.fragment(run_every="1s")
    def show_warning(processor):
        warning = processor.get_warning()
        if warning:
            st.error(warning)
        else:
            st.success(f"{max_distance_m:.1f}m 이내에 감지된 객체가 없습니다.")

        _SPEECH_ALERT(
            data={
                "text": warning,
                "repeatDelayMs": int(VOICE_WARNING_REPEAT_SEC * 1000),
            },
            key="distance-warning-speech",
        )

    show_warning(webrtc_ctx.video_processor)
# sad s