# PlantCare AI

## Обзор

**PlantCare AI** — сервис для диагностики болезней комнатных растений по фотографиям с предоставлением экологичных рекомендаций по уходу. Проект помогает выявлять заболевания и улучшать состояние комнатных растений с помощью современных методов компьютерного зрения и искусственного интеллекта.

## Основная структура репозитория
- **Обобщенные результаты/** - основная ифнормация с графиками моделей и кодом обучения.
- **plant_diseases/** — тренировочный датасет изображений для обучения моделей.
- **test_data/** — датасет изображений для тестирования моделей.
- **обновленный телешграм-бот/** — код для работы Телеграм бота.
- **metrics/** — файлы с метриками, расчетами и экспериментальными результатами.
- **predict.py** — скрипт для инференса (предсказания) по загруженной фотографии растения.
- **train_EfficientNet_B3_sphereface_split.py** — основной скрипт для обучения модели.
- **test_dataset_check.py** — скрипт для проверки модели на ткстовом датасете.
- **batch_visualization.png, biology-14-00099-v2.pdf** — визуализации и научные материалы по теме.
- **Консоль ...** — текстовые логи и отчёты по экспериментам (ArcFace, CosFace, точности, аугментации и др.).
- **Давние файлы/** - первые знакомства с обучением, пробы. 
- **README.md** — данный файл с описанием проекта.
- **.gitignore, .gitattributes** — служебные файлы для Git.

## История и развитие

- Проект прошёл несколько этапов: от работы с агрокультурными растениями до перехода на комнатные и теста разных моделей и лосс функций.
- Использовались различные подходы: TensorFlow, далее — PyTorch.
- Основные архитектуры: ArcFace, CosFace, SphereFace эксперименты с аугментацией и метриками.
- В коммитах отражены попытки улучшения точности и стабильности, переход между фреймворками, ведение метрик и логи экспериментов.

## Использование

Найдите бота в Telegram @plantcareaibot и узнайте чем болеет ваш комнатный питомец!

---

Разработано в Уральском федеральном университете.
