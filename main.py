from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from camera import create_hands_model, detect_hands, open_camera
from config import KEY_BACKSPACE, KEY_ESC, KEY_SPACE, MIN_CONFIDENCE, MODEL_PATH
from hand_features import FEATURE_SIZE, extract_landmarks

WINDOW_NAME = "Hand Sign Recognition"
MAX_HANDS = 2
LETTER_STABLE_FRAMES = 20
MAX_VISIBLE_TEXT = 42
USE_PINCH_FOR_SPACE = True

Classifier = Callable[[object, str], tuple[str, float]]


def _distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _palm_width(hand) -> float:
    lm = hand.landmark
    return max(0.001, abs(lm[5].x - lm[17].x))


def _extended_fingers(hand) -> list[bool]:
    lm = hand.landmark
    return [lm[tip].y < lm[joint].y for tip, joint in ((8, 6), (12, 10), (16, 14), (20, 18))]


def _is_thumb_extended(hand, handedness: str) -> bool:
    lm = hand.landmark
    if handedness == "Left":
        return lm[4].x > lm[3].x
    return lm[4].x < lm[3].x


def recognize_sign(hand, handedness: str) -> str:
    lm = hand.landmark
    index, middle, ring, pinky = _extended_fingers(hand)
    thumb = _is_thumb_extended(hand, handedness)
    palm_width = _palm_width(hand)

    thumb_index_touching = _distance(lm[4], lm[8]) < palm_width * 0.30
    index_middle_gap = abs(lm[8].x - lm[12].x) / palm_width

    if thumb_index_touching and middle and ring and pinky:
        return "F"
    if thumb and index and middle and pinky and not ring:
        return "I Love You"
    if thumb and index and not middle and not ring and not pinky:
        return "L"
    if thumb and pinky and not index and not middle and not ring:
        return "Y"
    if index and middle and not ring and not pinky:
        return "V" if index_middle_gap > 0.22 else "U"
    if index and middle and ring and not pinky:
        return "W"
    if index and not middle and not ring and not pinky:
        return "D"
    if pinky and not index and not middle and not ring:
        return "I"

    extended_count = sum((index, middle, ring, pinky))
    if extended_count == 4:
        return "B"
    if extended_count == 0:
        thumb_near_index = abs(lm[4].x - lm[5].x) / palm_width < 0.45
        return "A" if thumb and not thumb_near_index else "S"

    return "Unknown sign"


def heuristic_classifier(hand, handedness: str) -> tuple[str, float]:
    return recognize_sign(hand, handedness), 1.0


def load_trained_classifier(path: Path = MODEL_PATH) -> Optional[Classifier]:
    if not path.exists():
        print(f"Modelo {path} nao encontrado: usando regras heuristicas.")
        return None

    try:
        import joblib
    except ImportError:
        print("scikit-learn/joblib nao instalado: usando regras heuristicas.")
        return None

    bundle = joblib.load(path)
    model = bundle["model"]

    if bundle.get("feature_size") != FEATURE_SIZE:
        print("Modelo incompativel com hand_features.py: usando heuristicas.")
        return None

    classes = [str(c) for c in model.classes_]
    print(f"Modelo carregado ({len(classes)} letras): {' '.join(classes)}")

    def classify(hand, handedness: str) -> tuple[str, float]:
        features = extract_landmarks(hand, handedness).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])

        if confidence < MIN_CONFIDENCE:
            return "Unknown sign", confidence
        return classes[best], confidence

    return classify


def is_space_gesture(hand) -> bool:
    lm = hand.landmark
    others_curled = all(lm[tip].y > lm[joint].y for tip, joint in ((12, 10), (16, 14), (20, 18)))
    return _distance(lm[4], lm[8]) < _palm_width(hand) * 0.30 and others_curled


@dataclass
class WordBuilder:
    text: str = ""
    candidate: Optional[str] = None
    candidate_frames: int = 0
    latched: bool = False

    def release(self) -> None:
        self.candidate = None
        self.candidate_frames = 0
        self.latched = False

    def clear(self) -> None:
        self.text = ""
        self.release()

    def add_space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def observe(self, command: str) -> str:
        if not (command == "SPACE" or (len(command) == 1 and command.isalpha())):
            self.release()
            return "Mostre uma letra"

        if self.latched:
            return "Solte a pose entre letras"

        if command == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = command
            self.candidate_frames = 1

        if self.candidate_frames < LETTER_STABLE_FRAMES:
            return f"Mantenha {command}..."

        self.latched = True

        if command == "SPACE":
            self.add_space()
            return "Espaco inserido"

        self.text += command
        return f"Letra {command} adicionada"


def draw_hand_label(frame: np.ndarray, hand, label: str, sign: str) -> None:
    height, width = frame.shape[:2]
    wrist = hand.landmark[0]
    x = max(10, min(width - 260, int(wrist.x * width)))
    y = max(30, int(wrist.y * height) - 15)
    cv2.putText(frame, f"{label}: {sign}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)


def draw_interface(frame: np.ndarray, word_builder: WordBuilder, status: str) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (0, height - 82), (width, height), (0, 0, 0), -1)

    visible_text = word_builder.text[-MAX_VISIBLE_TEXT:]
    if len(word_builder.text) > MAX_VISIBLE_TEXT:
        visible_text = "..." + visible_text

    cv2.putText(frame, f"Texto: {visible_text or '_'}", (12, height - 43),
                font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{status} | Espaco/Backspace | duas maos: limpar",
                (12, height - 13), font, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "Q / Esc: sair", (10, 28), font, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)


def process_frame(frame, hands_model, word_builder: WordBuilder, clear_active: bool,
                  status: str, classify: Classifier) -> tuple[np.ndarray, bool, str]:
    frame, hands = detect_hands(frame, hands_model)

    if len(hands) == 2:
        if not clear_active:
            word_builder.clear()
            status = "Texto limpo"
        clear_active = True
        word_builder.release()

    elif hands:
        clear_active = False
        side, hand = hands[0]

        if USE_PINCH_FOR_SPACE and is_space_gesture(hand):
            sign, confidence = "SPACE", 1.0
        else:
            sign, confidence = classify(hand, side)

        command = sign if sign == "SPACE" or len(sign) == 1 else "UNKNOWN"
        status = word_builder.observe(command)
        draw_hand_label(frame, hand, side, f"{sign} ({confidence:.0%})")

    else:
        clear_active = False
        word_builder.observe("UNKNOWN")
        status = "Mostre uma letra"

    draw_interface(frame, word_builder, status)
    return frame, clear_active, status


def main() -> int:
    camera = open_camera()
    if camera is None:
        print("Nenhuma webcam funcional foi encontrada. Verifique as permissoes da camera.")
        return 1

    classify = load_trained_classifier() or heuristic_classifier
    word_builder = WordBuilder()
    clear_active = False
    status = "Mostre uma letra"

    try:
        with create_hands_model(max_hands=MAX_HANDS) as hands_model:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Nao foi possivel ler a webcam.")
                    break

                frame, clear_active, status = process_frame(
                    frame, hands_model, word_builder, clear_active, status, classify
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), KEY_ESC):
                    break
                if key == KEY_SPACE:
                    word_builder.add_space()
                    status = "Espaco inserido"
                elif key == KEY_BACKSPACE:
                    word_builder.backspace()
                    status = "Letra apagada"
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())