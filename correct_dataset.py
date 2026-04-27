import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import uuid

class DatasetCleaner:
    def __init__(self, root, input_dir="raw_dataset", output_dir="cleaned_dataset"):
        self.root = root
        self.root.title("Knitting Dataset Cleaner PRO - Safe Mode")
        self.root.geometry("1100x800") # Чуть расширили окно для новых кнопок
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.image_paths = []
        self.current_idx = 0
        self.current_img = None
        self.display_img = None
        self.photo = None
        
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.scale_ratio = 1.0

        # Словарь для хранения счетчиков сохраненных изображений по классам
        self.saved_counts = {}
        self.scan_existing_counts()

        self.setup_ui()
        self.load_dataset()

    def scan_existing_counts(self):
        """Сканирует папку output_dir при запуске, чтобы подтянуть текущие счетчики"""
        if not os.path.exists(self.output_dir):
            return
            
        for folder_name in os.listdir(self.output_dir):
            folder_path = os.path.join(self.output_dir, folder_name)
            if os.path.isdir(folder_path):
                # Считаем количество изображений в папке
                count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.saved_counts[folder_name] = count

    def setup_ui(self):
        self.info_label = tk.Label(self.root, text="Загрузка...", font=("Arial", 11, "bold"))
        self.info_label.pack(pady=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="#333333",
                                yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.vbar.config(command=self.canvas.yview)
        self.hbar.config(command=self.canvas.xview)
        
        # ROI 
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        # Добавили новые кнопки в интерфейс
        tk.Button(btn_frame, text="<< Назад", command=self.prev_image, width=10).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Пропустить (Del)", command=self.skip_image, bg="#ffcccc", width=15).grid(row=0, column=1, padx=5)
        
        tk.Button(btn_frame, text="↺ Поворот (Q)", command=lambda: self.rotate_image(90), width=12).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="Поворот ↻ (E)", command=lambda: self.rotate_image(-90), width=12).grid(row=0, column=3, padx=5)
        
        tk.Button(btn_frame, text="Сохранить ROI (Enter)", command=self.save_roi, bg="#ccffcc", width=20).grid(row=0, column=4, padx=5)
        tk.Button(btn_frame, text="Сохранить ЦЕЛИКОМ (S)", command=self.save_whole, bg="#cce5ff", width=22).grid(row=0, column=5, padx=5)
        tk.Button(btn_frame, text="Вперед >>", command=self.next_image, width=10).grid(row=0, column=6, padx=5)
        
        # Бинды клавиш
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<Delete>", lambda e: self.skip_image())
        self.root.bind("<Return>", lambda e: self.save_roi())
        self.root.bind("s", lambda e: self.save_whole())
        self.root.bind("S", lambda e: self.save_whole())
        # Бинды для поворота
        self.root.bind("q", lambda e: self.rotate_image(90))
        self.root.bind("Q", lambda e: self.rotate_image(90))
        self.root.bind("e", lambda e: self.rotate_image(-90))
        self.root.bind("E", lambda e: self.rotate_image(-90))

    def on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def load_dataset(self):
        for root_dir, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, file))
        
        if not self.image_paths:
            messagebox.showerror("Ошибка", f"В папке {self.input_dir} нет картинок!")
            self.root.destroy()
            return
            
        self.show_image()

    def rotate_image(self, angle):
        """Поворачивает текущее изображение и обновляет холст"""
        if self.current_img:
            # expand=True обязательно, чтобы картинка не обрезалась по краям при повороте
            self.current_img = self.current_img.rotate(angle, expand=True)
            self.refresh_display()

    def update_info_label(self):
        """Обновляет текст с прогрессом и счетчиком чистых картинок"""
        img_path = self.image_paths[self.current_idx]
        folder_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)
        
        # Получаем количество чистых изображений для текущего класса
        cleaned_count = self.saved_counts.get(folder_name, 0)
        
        self.info_label.config(
            text=f"Папка: [{folder_name}] | Оригинал: {file_name} | "
                 f"Прогресс: {self.current_idx + 1}/{len(self.image_paths)} | "
                 f"ОЧИЩЕНО В ЭТОМ КЛАССЕ: {cleaned_count}"
        )

    def show_image(self):
        if self.current_idx < 0: self.current_idx = 0
        if self.current_idx >= len(self.image_paths): self.current_idx = len(self.image_paths) - 1
            
        img_path = self.image_paths[self.current_idx]
        
        # Загружаем оригинал заново при переходе на картинку
        self.current_img = Image.open(img_path)
        self.refresh_display()

    def refresh_display(self):
        """Перерисовывает холст (вызывается при загрузке или после поворота)"""
        self.update_info_label()
  
        orig_w, orig_h = self.current_img.size
        max_w, max_h = 2000, 2000
        
        if orig_w > max_w or orig_h > max_h:
            self.scale_ratio = min(max_w/orig_w, max_h/orig_h)
        else:
            self.scale_ratio = 1.0
            
        new_w = int(orig_w * self.scale_ratio)
        new_h = int(orig_h * self.scale_ratio)
        
        self.display_img = self.current_img.resize((new_w, new_h), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.rect = None

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='#00ff00', width=3)

    def on_move_press(self, event):
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self.show_image()

    def prev_image(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_image()

    def skip_image(self):
        """Просто убирает картинку из текущего списка просмотра, не трогая файл на диске"""
        if not self.image_paths: return
        
        self.image_paths.pop(self.current_idx)
        
        if self.current_idx >= len(self.image_paths):
            self.current_idx -= 1
            
        if self.image_paths:
            self.show_image()
        else:
            messagebox.showinfo("Готово", "Список картинок пуст!")
            self.root.destroy()

    def generate_save_path(self, folder_name):
        unique_id = uuid.uuid4().hex[:8]
        new_filename = f"{unique_id}.jpg"
        
        save_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        return os.path.join(save_dir, new_filename)

    def increment_save_counter(self, folder_name):
        """Увеличивает счетчик для папки и обновляет UI"""
        if folder_name not in self.saved_counts:
            self.saved_counts[folder_name] = 0
        self.saved_counts[folder_name] += 1
        self.update_info_label()

    def save_roi(self):
        if not self.rect:
            messagebox.showwarning("Внимание", "Выделите область!")
            return
            
        coords = self.canvas.coords(self.rect)
        if len(coords) != 4: return
        
        x1, y1, x2, y2 = coords
        
        orig_x1 = int(min(x1, x2) / self.scale_ratio)
        orig_y1 = int(min(y1, y2) / self.scale_ratio)
        orig_x2 = int(max(x1, x2) / self.scale_ratio)
        orig_y2 = int(max(y1, y2) / self.scale_ratio)
        
        cropped_img = self.current_img.crop((orig_x1, orig_y1, orig_x2, orig_y2))
        
        folder_name = os.path.basename(os.path.dirname(self.image_paths[self.current_idx]))
        save_path = self.generate_save_path(folder_name)
        
        # Если картинка имеет альфа-канал после поворота, конвертируем в RGB для JPEG
        if cropped_img.mode in ("RGBA", "P"):
            cropped_img = cropped_img.convert("RGB")
            
        cropped_img.save(save_path)
        print(f"[ROI] Сохранено: {save_path}")
        
        self.increment_save_counter(folder_name)
        
        self.canvas.delete(self.rect)
        self.rect = None

    def save_whole(self):
        folder_name = os.path.basename(os.path.dirname(self.image_paths[self.current_idx]))
        save_path = self.generate_save_path(folder_name)
        
        img_to_save = self.current_img
        # Если картинка имеет альфа-канал после поворота, конвертируем в RGB для JPEG
        if img_to_save.mode in ("RGBA", "P"):
            img_to_save = img_to_save.convert("RGB")
            
        img_to_save.save(save_path)
        print(f"[ЦЕЛИКОМ] Сохранено: {save_path}")
        
        self.increment_save_counter(folder_name)
        self.next_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCleaner(root, input_dir="raw_dataset", output_dir="cleaned_dataset")
    root.mainloop()