import os
import cv2
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict
from facenet_pytorch import InceptionResnetV1
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis
from app.database import add_employee as db_add_employee, log_attendance
from array import array

class FaceRecognizer:
    def __init__(self, database_path, similarity_threshold=0.7,
                 min_face_size=50, max_faces=10, face_size=(112, 112)):
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.max_faces = max_faces
        self.face_size = face_size
        self.database_path = database_path  # сохраняем путь к базе лиц

        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))

        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

        self.face_db = self.load_face_database(self.database_path)

    def load_face_database(self, database_path=None):
        if database_path is None:
            database_path = self.database_path

        face_db = defaultdict(list)
        for person_name in os.listdir(database_path):
            person_dir = os.path.join(database_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    faces = self.face_app.get(img)
                    for face in faces:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                            continue
                        face_img = img[y1:y2, x1:x2]
                        embedding = self.get_face_embedding_batch([face_img])[0]
                        face_db[person_name].append(embedding)
                except Exception as e:
                    print(f"[WARN] Ошибка обработки {img_path}: {e}")

        processed_db = {}
        for name, embeddings in face_db.items():
            if embeddings:
                arr = np.vstack(embeddings)
                processed_db[name] = {
                    'mean_embedding': np.mean(arr, axis=0),
                    'embeddings': arr,
                    'variance': np.var(arr, axis=0)
                }
                print(f"[INFO] Загружено {len(arr)} фото для {name} (дисперсия: {np.mean(processed_db[name]['variance']):.4f})")
        self.face_db = processed_db
        return processed_db

    def get_face_embedding_batch(self, faces):
        tensors = []
        for face in faces:
            face = cv2.resize(face, self.face_size)

            # 1. Извлекаем сырые байты из numpy массива (OpenCV)
            # 2. Создаем стандартный массив байтов Python без вызова NumPy API
            raw_data = array('B', face.tobytes())

            # 3. PyTorch создает тензор напрямую из встроенного типа Python
            h, w, c = face.shape
            face_tensor = torch.tensor(raw_data, dtype=torch.uint8).reshape(h, w, c)

            # 4. Меняем оси и нормализуем
            face_tensor = face_tensor.permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            tensors.append(face_tensor)

        device = next(self.face_recognizer.parameters()).device
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            output = self.face_recognizer(batch).cpu()
            # Возвращаем список Python вместо numpy массива
            return output.tolist()

    def recognize_faces(self, frame):
        faces = self.face_app.get(frame)
        recognitions = []

        face_imgs = []
        boxes = []
        for i, face in enumerate(faces):
            if i >= self.max_faces:
                break
            x1, y1, x2, y2 = map(int, face.bbox)
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue
            face_imgs.append(frame[y1:y2, x1:x2])
            boxes.append((x1, y1, x2, y2))

        if not face_imgs:
            return []

        embeddings = self.get_face_embedding_batch(face_imgs)
        for i, emb in enumerate(embeddings):
            best_match, best_score = None, 1.0
            for name, data in self.face_db.items():
                for db_emb in data['embeddings']:
                    dist = cosine(emb, db_emb)
                    if dist < best_score:
                        best_score = dist
                        best_match = name

            is_known = best_score < (1 - self.similarity_threshold)
            recognitions.append({
                'box': boxes[i],
                'name': best_match if is_known else None,
                'similarity': 1 - best_score,
            })
        return recognitions

    def draw_results(self, frame, recognitions):
        for rec in recognitions:
            x1, y1, x2, y2 = rec['box']
            is_known = rec['name'] is not None
            # первый цвет зёленый, если нашёл совпадение, красный в другом случаи. Формат
            color = (0, 255, 0) if is_known else (0, 0, 255)
            label = rec['name'] if is_known else 'Unknown'
            label += f": {round(rec['similarity'] * 100, 2)}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def register_employees_to_db(self):
        for name in self.face_db.keys():
            db_add_employee(name)

    def add_employee(self, name, img_path):
        import shutil
        # Путь к базе лиц (например, можно менять на свой)
        base_dir = self.database_path
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Создаем папку для сотрудника, если нужна
        person_dir = os.path.join(base_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        ext = os.path.splitext(img_path)[1]
        # Имя файла уникальное (например, с меткой времени)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ext
        dest_path = os.path.join(person_dir, filename)

        shutil.copyfile(img_path, dest_path)

        # Добавить сотрудника в базу сотрудников (если нужно)
        db_add_employee(name)

        # Перезагрузить базу лиц с учётом нового фото
        self.load_face_database()
