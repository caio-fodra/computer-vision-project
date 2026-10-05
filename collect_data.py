from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2

from camera import create_hands_model, detect_hands, open_camera
from config import DATA_PATH, KEY_BACKSPACE, KEY_ESC
from hand_features import FEATURE_SIZE, extract_landmarks

SAMPLES_PER_BURST = 150
COUNTDOWN_FRAMES = 45


def load_counts(path: Path) -> Counter:
    counts: Counter = Counter()

    if not path.exists():
        return counts

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if row:
                counts[row[0]] += 1

    return counts


def open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    file = path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(file)

    if is_new:
        writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_SIZE)])

    return file, writer


def draw_overlay(frame, counts: Counter, message: str) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (0, 0), (width, 70), (0, 0, 0), -1)
    cv2.putText(frame, message, (10, 28), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Tecla A-Z: gravar | Backspace: cancelar | Esc: sair",
                (10, 58), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, summary or "Nenhum exemplo gravado ainda",
                (10, height - 10), font, 0.45, (180, 255, 180), 1, cv2.LINE_AA)


def main() -> int:
    camera = open_camera()
    if camera is None:
        print("Nenhuma webcam funcional foi encontrada.")
        return 1

    counts = load_counts(DATA_PATH)
    file, writer = open_writer(DATA_PATH)

    current_label: Optional[str] = None
    countdown = 0
    recorded = 0
    message = "Faca o sinal e aperte a tecla da letra"

    try:
        with create_hands_model(max_hands=1) as hands_model:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Nao foi possivel ler a webcam.")
                    break

                frame, hands = detect_hands(frame, hands_model)

                if current_label is not None:
                    if countdown > 0:
                        countdown -= 1
                        message = f"Prepare {current_label}... {countdown // 15 + 1}"
                    elif len(hands) == 1:
                        side, hand = hands[0]
                        writer.writerow([current_label] + extract_landmarks(hand, side).tolist())
                        recorded += 1
                        counts[current_label] += 1
                        message = f"Gravando {current_label}: {recorded}/{SAMPLES_PER_BURST}"

                        if recorded >= SAMPLES_PER_BURST:
                            file.flush()
                            message = f"{current_label} concluida. Proxima letra?"
                            current_label = None
                    else:
                        message = f"Gravando {current_label}: mao nao detectada"

                draw_overlay(frame, counts, message)
                cv2.imshow("Coleta de sinais", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == KEY_ESC:
                    break

                if key == KEY_BACKSPACE and current_label is not None:
                    message = f"Gravacao de {current_label} cancelada ({recorded} frames ja salvos)"
                    current_label = None
                    file.flush()
                    continue

                if current_label is None and key != 255:
                    char = chr(key).upper()
                    if "A" <= char <= "Z":
                        current_label = char
                        countdown = COUNTDOWN_FRAMES
                        recorded = 0
    finally:
        file.close()
        camera.release()
        cv2.destroyAllWindows()

    print(f"Dados salvos em {DATA_PATH}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())