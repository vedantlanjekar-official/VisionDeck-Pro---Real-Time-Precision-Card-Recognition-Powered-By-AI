import streamlit as st
import cv2
from utils.card_game_detector import CardGameDetector
from utils.constants import MODEL_PATH, CLASS_NAMES
from utils.text_constants import Texts


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "detected_cards" not in st.session_state:
        st.session_state.detected_cards = []
    if "texts" not in st.session_state:
        st.session_state.texts = Texts()


def capture_cards(detector):
    """Capture cards using the webcam and process detections."""
    texts = st.session_state.texts
    st.write(texts.get("capturing_cards"))
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    frame_placeholder = st.empty()
    detected_classes = []
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frame_placeholder.image(frame, channels="BGR")
            # Detect on the frame we already read (fixes double-read issue)
            detected_classes.extend(detector.detect_on_frame(frame))
    cap.release()
    frame_placeholder.empty()

    detections = detector.aggregate_detections(detected_classes)
    detected_cards = detector.parse_cards(detections)

    if detected_cards:
        st.success(texts.get("cards_detected"))
        st.session_state.detected_cards = detected_cards
    else:
        st.error(texts.get("no_cards_detected"))


def display_results():
    """Display the detected cards."""
    texts = st.session_state.texts
    
    if st.session_state.detected_cards:
        st.subheader(texts.get("detected_cards"))
        card_strings = [str(card) for card in st.session_state.detected_cards]
        st.write(", ".join(card_strings))
        st.write(f"**{texts.get('card_count')} {len(st.session_state.detected_cards)}**")
    else:
        st.info("No cards detected yet. Click 'Take Snapshot' to detect cards.")


def main():
    initialize_session_state()
    texts = st.session_state.texts
    detector = CardGameDetector(MODEL_PATH, CLASS_NAMES)

    st.set_page_config(page_title=texts.get("page_title"), layout="wide")
    st.title(texts.get("title"))
    st.subheader(texts.get("subtitle"))
    
    st.markdown("---")

    # Buttons section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(texts.get("take_snapshot"), use_container_width=True, type="primary"):
                capture_cards(detector)
                st.rerun()

    with col2:
        if st.button(texts.get("clear_results"), use_container_width=True):
            st.session_state.detected_cards = []
            st.success("Results cleared!")
                st.rerun()

        st.markdown("---")

    # Display results automatically if cards are detected
    if st.session_state.detected_cards:
        display_results()
    else:
        st.info("No cards detected yet. Click 'Take Snapshot' to detect cards from your webcam.")


if __name__ == "__main__":
    main()
