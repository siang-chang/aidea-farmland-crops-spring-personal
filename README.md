# AIdea: Farmland and Crops - Sprint

[AIdea 農地作物現況調查影像辨識競賽 - 春季賽：AI 作物影像判釋](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340?focus=intro)

這個競賽與 [@Tsao666](https://github.com/Tsao666) 及 [@Tianming8585](https://github.com/Tianming8585) 共同參賽，我們的最終排名為 30/151

## 簡介

農地作物現況調查可使用人員搭配相機於現地拍照紀錄，然而農地區域廣泛、我國之坵塊分割細碎且分佈零碎，所獲取之照片影像資料龐大，轉換為可用於管理操作之資訊，極度耗費人力、時間。AI 技術對於影像判識工作近年已有長足之進步，適合導入農地作物現況調查之工作程序  ，加速農政單位獲取相關資訊

目前雖然有完整的民生/工業/醫療等 AI 資料集，但在農業數據相對缺乏，故在未來 AI 智慧農業需求上，將需要投入大量的專業人力進行農業數據蒐集及分析作業；透過本競賽，將協助學生瞭解農業資料集及農產業影像辨識的應用需求，並培育學生應用 AI 進行農業領域影像辨識的經驗及技術能力

本競賽資料來源為行政院農委會農業試驗所，影像均已由專家進行分類，包含 13 種作物及 2 種農地狀態之標註資訊。參賽隊伍需能使用 AI 正確辨識各影像之作物

評分標準：Weighted-Precision，且各類別的 F1-score 需大於 0.7

議題提供單位：成功大學測量及空間資訊學系

## 活動時間

| 時間            | 事件                                          |
| --------------- | --------------------------------------------- |
| 2022/3/14       | 開放報名及下載訓練資料集                      |
| 2022/5/14 00:00 | 報名截止，開放下載測試資料集及上傳答案        |
| 2022/5/15 23:59 | 關閉上傳答案                                  |
| 2022/5/17 14:00 | 公布成績，開始上傳報告及程式碼                |
| 2022/5/25 23:59 | 截止上傳報告及程式碼，開始評估(程式碼 + 報告) |
| 2022/6/17 14:00 | 公布最終結果                                  |
| Mar-23          | 頒獎典禮 (暫定)                               |

## Predictions on testset

| method         | prediction link                                                                                                           |
| :------------- | ------------------------------------------------------------------------------------------------------------------------- |
| DCNN(DOY)      | [link.pkl](./src/outputs/fullset/dcnn-doy-3-conv12-heuniform-adam-labelsmoothing-classweight-test.pkl)                    |
| DCNN(DOY, Sig) | [link](./src/outputs/fullset/dcnn-doy-1-sigmoid-conv12-heuniform-l2norm-adam-labelsmoothing-classweight-test.pkl)         |
| DCNN(w/o DOY)  | [link](./src/outputs/fullset/dcnn-doy-sod-3-conv14-heuniform-adam-labelsmoothing-classweight-test.pkl)                    |
| DCNN(DOY, SOD) | [link](./src/outputs/fullset/dcnn-conv14-heuniform-adam-labelsmoothing-classweight-test.pkl)                              |
| ViT            | [link](./src/outputs/fullset/vision-transformer-result-test.pkl)                                                          |
| EfficientNetB6 | [link](./src/outputs/fullset/efficientnetb6-transfer-trainable-epochs150-batchsize64-lrreducer-sampleweight-224-test.pkl) |
| EfficientNetB3 | [link](./src/outputs/fullset/efficientnetb3-transfer-trainable-epochs150-batchsize32-lrreducer-sampleweight-224-test.pkl) |

## Predictions on fullset

| method         |  score | prediction link                                                                                                            |
| :------------- | -----: | :------------------------------------------------------------------------------------------------------------------------- |
| DCNN(DOY)      | 0.9776 | [link.pkl](./src/outputs/fullset/dcnn-doy-3-conv12-heuniform-adam-labelsmoothing-classweight-train.pkl)                    |
| DCNN(DOY, Sig) | 0.9750 | [link](./src/outputs/fullset/dcnn-doy-1-sigmoid-conv12-heuniform-l2norm-adam-labelsmoothing-classweight-train.pkl)         |
| DCNN(w/o DOY)  | 0.9757 | [link](./src/outputs/fullset/dcnn-doy-sod-3-conv14-heuniform-adam-labelsmoothing-classweight-train.pkl)                    |
| ViT            | 0.9805 | [link](./src/outputs/fullset/vision-transformer-result-train.pkl)                                                          |
| EfficientNetB6 | 0.9782 | [link](./src/outputs/fullset/efficientnetb6-transfer-trainable-epochs150-batchsize64-lrreducer-sampleweight-224-train.pkl) |
| EfficientNetB3 | 0.9842 | [link](./src/outputs/fullset/efficientnetb3-transfer-trainable-epochs150-batchsize32-lrreducer-sampleweight-224-train.pkl) |
