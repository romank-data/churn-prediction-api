# Churn Prediction API

## Описание
Проект реализует модель предсказания оттока пользователей на основе данных об онлайн-играх и сундуках.  
Модель обучена с помощью LightGBM и оформлена в виде Python-пайплайна.  
Для API используется FastAPI.

## Требования
- Python 3.8+
- Зависимости перечислены в `requirements.txt`

## Установка

pip install -r requirements.txt

## Запуск сервера
FastAPI сервер через uvicorn командой:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

## API

### POST/predict

- Принимает JSON с двумя списками объектов:
  - `games`: данные по играм
  - `chests`: данные по сундукам
- Возвращает вероятности оттока по каждому игроку

Пример запроса:

{
  "games": [
    {
      "_id": "68cc1f55586c34cb4a03f569",
      "game_mode": "15RedFrame",
      "creator_id": "68cbffd1ba2c0149750a8263",
      "users": [
        {
          "_id": "68cbffd1ba2c0149750a8263",
          "username": "king1567asdf",
          "created_at": "2025-09-18T12:49:21.608000Z",
          "seconds_in_game": 15424.0,
          "online": {
            "online_sessions": 2.0
          },
          "online_game_rating": {
            "value": 1427.0
          },
          "energy": {
            "count": 322
          }
        },
        {
          "_id": "68ca93f6f85da64f410c8f84",
          "username": "kanadeharuka",
          "created_at": "2025-09-17T10:56:54.720000Z",
          "seconds_in_game": 30261.0,
          "online": {
            "online_sessions": 19.0
          },
          "online_game_rating": {
            "value": 1697.0
          },
          "energy": {
            "count": 1401.0
          }
        }
      ],
      "status": 2,
      "started_at": 1758207847.0,
      "ended_at": 1758209130.0,
      "winner": "68ca93f6f85da64f410c8f84",
      "score": [
        0,
        1
      ],
      "frames_count": 1,
      "isRematch": null,
      "updated_at": "2025-09-18T15:25:30.890000Z",
      "created_at": "2025-09-18T15:03:49.216000Z",
      "end_stats": {
        "rating_points": [
          -15.0,
          14.0
        ],
        "highest_break": [
          8.0,
          5.0
        ],
        "balls_potted": [
          10.0,
          12.0
        ],
        "total_points": [
          30.0,
          60.0
        ],
        "table_time": [
          0.5265625,
          0.4734375
        ],
        "pot_success": [
          0.3030303030303,
          0.35294117647059
        ],
        "shot_time": [
          14242.424242424,
          11911.764705882
        ],
        "game_id": "68cc1f55586c34cb4a03f569",
        "updated_at": "2025-09-18T15:25:30.983000Z",
        "created_at": "2025-09-18T15:25:30.983000Z"
      }
    }
  ],
  "chests": [
    {
      "chest": {
        "type": "daily"
      },
      "user": {
        "_id": "68cbffd1ba2c0149750a8263",
        "username": "king1567asdf"
      },
      "opened_with": "time",
      "open_at": 1758207817
    }
  ]
}

Пример ответа:
{
    "probabilities": [
        0.545240992209984,
        0.3657868856502453
    ]
}

## Структура проекта
- `train.py` — обучение и тестирование модели
- `pipeline.py` — пайплайн препроцессинга и модель
- `preprocess.py` — препроцессинг данных
- `utils.py` — вспомогательные функции
- `main.py` — FastAPI сервис для запуска модели
- `requirements.txt` — зависимости
- `get_json.py` — вытягиваем рандомный json для проверки
- `nickname_prediction.py` - предсказания по нику
- `random_prediction.py` - предсказание на рандомном юзере
- `all_predict.py` — предсказания на локальных csv

## Дополнительно
- Модель хранится в `churn_pipeline.pkl` — загружается FastAPI сервисом
- Для разработки включена автоперезагрузка сервера через `--reload`
- Для теста API перейдите в `http://localhost:8000/docs` после запуска сервера
