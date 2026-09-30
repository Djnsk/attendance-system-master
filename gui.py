import sys
import time
import shutil
import os
import csv
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout,
                             QFileDialog, QInputDialog, QMessageBox, QTableWidget, QTableWidgetItem,
                             QDialog, QHeaderView, QSizePolicy, QGridLayout)
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QSplitter
from PyQt6.QtCore import Qt
import cv2
from app.face_recognition_module import FaceRecognizer
from app.database import get_attendance_records, get_employees, log_attendance, delete_employee
from pathlib import Path


class RecognitionThread(QThread):
    result_ready = pyqtSignal(list)

    def __init__(self, recognizer, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            if self.frame is not None:
                result = self.recognizer.recognize_faces(self.frame)
                self.result_ready.emit(result)
                self.frame = None

    def submit_frame(self, frame):
        self.frame = frame.copy()

    def stop(self):
        self.running = False
        self.wait()


class HistoryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("История посещаемости")
        self.resize(600, 400)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["ID", "Имя", "Время посещения"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.load_data()

    def load_data(self):
        records = get_attendance_records()
        self.table.setRowCount(len(records))
        for row, (rec_id, name, timestamp) in enumerate(records):
            self.table.setItem(row, 0, QTableWidgetItem(str(rec_id)))
            self.table.setItem(row, 1, QTableWidgetItem(name))
            self.table.setItem(row, 2, QTableWidgetItem(timestamp))


class StatisticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Статистика посещаемости")
        self.resize(600, 400)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Имя", "Количество посещений"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.export_btn = QPushButton("Экспорт в CSV")
        self.export_btn.clicked.connect(self.export_csv)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.export_btn)
        self.setLayout(layout)

        self.load_stats()

    def load_stats(self):
        records = get_attendance_records()
        count_map = {}
        for _, name, _ in records:
            count_map[name] = count_map.get(name, 0) + 1

        self.table.setRowCount(len(count_map))
        for row, (name, count) in enumerate(count_map.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(count)))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Имя", "Количество посещений"])
            for row in range(self.table.rowCount()):
                name = self.table.item(row, 0).text()
                count = self.table.item(row, 1).text()
                writer.writerow([name, count])

        QMessageBox.information(self, "Экспорт", f"Данные успешно сохранены в {path}")

class LoadDatabaseThread(QThread):
    finished = pyqtSignal()

    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer

    def run(self):
        self.recognizer.face_db = self.recognizer.load_face_database()
        self.finished.emit()


class AttendanceApp(QWidget):
    def __init__(self, face_root_path):
        super().__init__()
        # --- Применение обновлённого стиля ---
        self.setStyleSheet("""
            /* Главная область */
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 13px;
            }

            /* Область видеопотока */
            QLabel#imageLabel {
                background-color: #020617;
                border: 2px solid #1e293b;
                border-radius: 12px;
            }

            /* Индикатор статуса */
            QLabel#statusLabel {
                color: #94a3b8;
                font-weight: 600;
                font-size: 14px;
                padding: 4px;
            }

            /* Заголовки секций */
            QLabel.sectionHeader {
                color: #38bdf8;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: 10px;
            }

            /* Базовые кнопки */
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #334155;
                border-color: #475569;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton:disabled {
                background-color: #0f172a;
                color: #475569;
                border-color: #1e293b;
            }

            /* Акцентная кнопка (Старт/Стоп распознавания) */
            QPushButton#heroButton {
                background-color: #2563eb;
                color: #ffffff;
                font-weight: 600;
                font-size: 15px;
                border: none;
                padding: 12px 18px;
            }
            QPushButton#heroButton:hover {
                background-color: #3d7afe;
            }
            QPushButton#heroButton:pressed {
                background-color: #1d4ed8;
            }

            /* Опасная кнопка (Удаление) */
            QPushButton#dangerButton {
                background-color: #451a03;
                color: #fca5a5;
                border-color: #7f1d1d;
            }
            QPushButton#dangerButton:hover {
                background-color: #991b1b;
                color: #ffffff;
            }

            /* Сплиттер */
            QSplitter::handle {
                background-color: #1e293b;
                width: 2px;
            }
        """)

        # --- Инициализация логических объектов ---
        self.cap = cv2.VideoCapture(0)
        self.last_recog_time = 0
        self.recognizer = FaceRecognizer(database_path=face_root_path)
        self.thread = RecognitionThread(self.recognizer)
        self.thread.result_ready.connect(self.handle_recognition_result)
        self.thread.start()

        self.last_recognitions = []
        self.frame_idx = 0
        self.recognition_running = False
        self.recording = False
        self.video_writer = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.last_seen = {}

        # --- Настройка визуальных элементов ---
        self.image_label = QLabel()
        self.image_label.setObjectName("imageLabel")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Статус: ожидание")
        self.status_label.setObjectName("statusLabel")

        # Кнопки управления
        self.recognition_button = QPushButton("Старт распознавания")
        self.recognition_button.setObjectName("heroButton")
        self.recognition_button.clicked.connect(self.toggle_recognition)

        self.add_employee_btn = QPushButton("Добавить студента")
        self.add_employee_btn.clicked.connect(self.add_employee)

        self.delete_employee_btn = QPushButton("Удалить студента")
        self.delete_employee_btn.setObjectName("dangerButton")
        self.delete_employee_btn.clicked.connect(self.delete_employee)

        self.show_history_btn = QPushButton("История визитов")
        self.show_history_btn.clicked.connect(self.show_history)

        self.show_stats_btn = QPushButton("Статистика")
        self.show_stats_btn.clicked.connect(self.show_statistics)

        self.record_video_btn = QPushButton("Запись видео")
        self.record_video_btn.clicked.connect(self.toggle_recording)

        self.take_photo_btn = QPushButton("Снимок экрана")
        self.take_photo_btn.clicked.connect(self.take_photo)

        self.refresh_db_btn = QPushButton("Обновить базу")
        self.refresh_db_btn.clicked.connect(self.refresh_face_database)

        # --- Компоновка панели управления ---
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(16, 16, 16, 16)
        control_layout.setSpacing(12)

        # Блок главного действия
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.recognition_button)

        # Секция: Студенты
        student_lbl = QLabel("Управление базой")
        student_lbl.setProperty("class", "sectionHeader")
        control_layout.addWidget(student_lbl)

        students_grid = QGridLayout()
        students_grid.setSpacing(8)
        students_grid.addWidget(self.add_employee_btn, 0, 0)
        students_grid.addWidget(self.delete_employee_btn, 0, 1)
        students_grid.addWidget(self.refresh_db_btn, 1, 0, 1, 2)
        control_layout.addLayout(students_grid)

        # Секция: Аналитика
        analytics_lbl = QLabel("Аналитика")
        analytics_lbl.setProperty("class", "sectionHeader")
        control_layout.addWidget(analytics_lbl)

        analytics_grid = QGridLayout()
        analytics_grid.setSpacing(8)
        analytics_grid.addWidget(self.show_history_btn, 0, 0)
        analytics_grid.addWidget(self.show_stats_btn, 0, 1)
        control_layout.addLayout(analytics_grid)

        # Секция: Захват медиа
        media_lbl = QLabel("Медиа")
        media_lbl.setProperty("class", "sectionHeader")
        control_layout.addWidget(media_lbl)

        media_grid = QGridLayout()
        media_grid.setSpacing(8)
        media_grid.addWidget(self.take_photo_btn, 0, 0)
        media_grid.addWidget(self.record_video_btn, 0, 1)
        control_layout.addLayout(media_grid)

        control_layout.addStretch()

        # Обертка панели управления
        control_panel = QWidget()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(380)

        # Основной Сплиттер
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setContentsMargins(12, 12, 12, 12)
        splitter.addWidget(self.image_label)
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    def handle_recognition_result(self, result):
        if not self.recognition_running:
            return
        self.last_recognitions = result
        current_time = time.time()
        for rec in result:
            name = rec.get('name')
            if not name:
                continue

            # если не было видно раньше или прошло > N секунд
            last_time = self.last_seen.get(name)
            if not last_time or current_time - last_time > 60:  # 60 сек = новое посещение
                log_attendance(name)
                self.last_seen[name] = current_time

    def toggle_recognition(self):
        if not self.recognition_running:
            self.timer.start(30)
            self.recognition_running = True
            self.recognition_button.setText("Остановить распознавание")
            self.status_label.setText("Статус: распознавание запущено")
        else:
            self.recognition_running = False
            self.last_recognitions = []
            self.recognition_button.setText("Старт распознавания")
            self.status_label.setText("Статус: распознавание остановлено")
            self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Ошибка захвата видео")
            return

        current_time = time.time()
        if self.recognition_running and current_time - self.last_recog_time > 1.0:
            self.last_recog_time = current_time
            self.thread.submit_frame(frame.copy())

        frame = self.recognizer.draw_results(frame, self.last_recognitions)

        # Запись видео, если включено
        if self.recording and self.video_writer:
            self.video_writer.write(frame)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

        self.frame_idx += 1

    def add_employee(self):
        name, ok = QInputDialog.getText(self, "Добавить сотрудника", "Введите имя сотрудника:")
        if not ok or not name.strip():
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите фото сотрудника", "", "Images (*.png *.jpg *.jpeg)")

        if not files:
            return

        try:
            person_dir = os.path.join(self.recognizer.database_path, name.strip())
            os.makedirs(person_dir, exist_ok=True)

            for i, src_path in enumerate(files, start=1):
                ext = os.path.splitext(src_path)[1]
                dest_path = os.path.join(person_dir, f"{i}{ext}")
                shutil.copyfile(src_path, dest_path)

            from app.database import add_employee as db_add_employee
            db_add_employee(name.strip())

            # 🧵 запускаем асинхронную загрузку базы
            self.status_label.setText("Статус: обновление базы лиц...")
            self.db_thread = LoadDatabaseThread(self.recognizer)
            self.db_thread.finished.connect(lambda: self.on_database_loaded(name, len(files)))
            self.db_thread.start()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при добавлении:\n{e}")

    def on_database_loaded(self, name, photo_count):
        QMessageBox.information(self, "Успех", f"Сотрудник {name} добавлен с {photo_count} фото.\nБаза лиц обновлена.")
        self.status_label.setText("Статус: ожидание")

    def on_face_db_refreshed(self):
        QMessageBox.information(self, "Обновление", "База лиц успешно обновлена.")
        self.status_label.setText("Статус: ожидание")

    def delete_employee(self):
        employees = get_employees()
        if not employees:
            QMessageBox.information(self, "Удаление", "Список сотрудников пуст.")
            return

        items = [e[1] for e in employees]
        name, ok = QInputDialog.getItem(self, "Удалить сотрудника", "Выберите сотрудника:", items, editable=False)
        if not ok or not name:
            return

        confirm = QMessageBox.question(self, "Подтверждение", f"Удалить сотрудника {name} из базы?")
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                delete_employee(name)
                QMessageBox.information(self, "Успех", f"Сотрудник {name} удалён.")
                self.refresh_face_database()
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось удалить сотрудника:\n{e}")

    def show_history(self):
        dialog = HistoryDialog(self)
        dialog.exec()

    def show_statistics(self):
        dialog = StatisticsDialog(self)
        dialog.exec()

    def toggle_recording(self):
        if not self.recording:
            # Создаём папку videos, если её нет
            videos_dir = os.path.join(os.getcwd(), 'videos')
            os.makedirs(videos_dir, exist_ok=True)

            # Формируем имя файла с датой и временем
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
            save_path = os.path.join(videos_dir, filename)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 20.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            if not self.video_writer.isOpened():
                QMessageBox.warning(self, "Ошибка", "Не удалось открыть файл для записи.")
                self.video_writer = None
                return
            self.recording = True
            self.record_video_btn.setText("Остановить запись видео")
            self.status_label.setText("Статус: запись видео")
        else:
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            self.record_video_btn.setText("Начать запись видео")
            self.status_label.setText("Статус: запись видео остановлена")

    def take_photo(self):
        ret, frame = self.cap.read()
        if not ret:
            QMessageBox.warning(self, "Ошибка", "Не удалось сделать снимок.")
            return

        # Создаём папку photos, если её нет
        photos_dir = os.path.join(os.getcwd(), 'photos')
        os.makedirs(photos_dir, exist_ok=True)

        # Формируем имя файла с датой и временем
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
        filepath = os.path.join(photos_dir, filename)

        cv2.imwrite(filepath, frame)
        QMessageBox.information(self, "Снимок", f"Фото сохранено в {filepath}")

    def refresh_face_database(self):
        self.status_label.setText("Статус: обновление базы лиц...")
        self.db_thread = LoadDatabaseThread(self.recognizer)
        self.db_thread.finished.connect(self.on_face_db_refreshed)
        self.db_thread.start()

    def closeEvent(self, event):
        self.timer.stop()
        self.thread.stop()
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        event.accept()


def create_face_folder():
    root_path = Path(__file__).parent / 'app' / 'face_db'

    root_path.mkdir(parents=True, exist_ok=True)

    return root_path


if __name__ == "__main__":
    face_db_root_path = create_face_folder()
    app = QApplication(sys.argv)
    window = AttendanceApp(face_root_path=face_db_root_path)
    window.show()
    sys.exit(app.exec())
