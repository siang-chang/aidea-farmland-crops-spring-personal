# AIdea: Farmland and Crops - Sprint

[AIdea 農地作物現況調查影像辨識競賽 - 春季賽：AI 作物影像判釋](https://aidea-web.tw/topic/93c8c26b-0e96-44bc-9a53-1c96353ad340?focus=intro)

這個競賽與 [@Tsao666](https://github.com/Tsao666) 及 [@Tianming8585](https://github.com/Tianming8585) 共同參賽，我們的最終排名為 30/151

## 簡介

農地作物現況調查可使用人員搭配相機於現地拍照紀錄，然而農地區域廣泛、我國之坵塊分割細碎且分佈零碎，所獲取之照片影像資料龐大，轉換為可用於管理操作之資訊，極度耗費人力、時間。AI 技術對於影像判識工作近年已有長足之進步，適合導入農地作物現況調查之工作程序 ，加速農政單位獲取相關資訊

目前雖然有完整的民生/工業/醫療等 AI 資料集，但在農業數據相對缺乏，故在未來AI智慧農業需求上，將需要投入大量的專業人力進行農業數據蒐集及分析作業；透過本競賽，將協助學生瞭解農業資料集及農產業影像辨識的應用需求，並培育學生應用 AI 進行農業領域影像辨識的經驗及技術能力

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

| Model                 | Prediction link                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :-------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DCNN(DOY)             | [DCNNDOY-3-conv12-he_uniform-adam-labelSmoothing-fullset-classWeight](src/evaluate/DCNN/DCNNDOY-3-conv12-he_uniform-adam-labelSmoothing-fullset-classWeight-testingSet.pkl)                                                                                                                                                                                                                                                                                                                                                                                                            |
| DCNN(DOY, Sigmoid)    | [DCNNDOY-1-sigmoid-conv12-he_uniform-l2norm-adam-labelSmoothing-fullset-classWeight](src/evaluate/DCNN/DCNNDOY-1-sigmoid-conv12-he_uniform-l2norm-adam-labelSmoothing-fullset-classWeight-testingSet.pkl)                                                                                                                                                                                                                                                                                                                                                                              |
| DCNN(w/o DOY)         | [DCNNDOYSOD-3-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight](src/evaluate/DCNN/DCNNDOYSOD-3-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight-testingSet.pkl)                                                                                                                                                                                                                                                                                                                                                                                                      |
| DCNN(DOY, SOD)        | [DCNN-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight](src/evaluate/DCNN/DCNN-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight-testingSet.pkl)                                                                                                                                                                                                                                                                                                                                                                                                                      |
| FaceChannel(DOY, SOD) | [FaceChannelDOYSOD-he_uniform-adamax-labelSmoothing-fullset-classWeight.pkl](src/evaluate/FaceChannel/FaceChannelDOYSOD-he_uniform-adamax-labelSmoothing-fullset-classWeight-testingSet.pkl)                                                                                                                                                                                                                                                                                                                                                                                           |
| ViT                   | [vision_transformer/ViT-default-20220509/predict_testset.pkl](https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/vision_transformer/ViT-default-20220509/predict_testset.pkl)                                                                                                                                                                                                                                                                                                                                                                                        |
| EfficientNet(B6)      | [efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224-test.csv](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224-test.csv>)<br>[efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224-test.pkl](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224-test.pkl>) |
| EfficientNet(B3)      | [efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224-test.csv](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224-test.csv>)<br>[efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224-test.pkl](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224-test.pkl>) |

## Predictions on fullset

| Model            | Weighed precision | Prediction link                                                                                                                                                                                                                                                                 |
| :--------------- | ----------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DCNN             |            0.9776 | [DCNN/DCNNDOY-3-conv12-he_uniform-adam-labelSmoothing-fullset-classWeight.pkl](https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/DCNN/DCNNDOY-3-conv12-he_uniform-adam-labelSmoothing-fullset-classWeight.pkl)                                               |
| DCNN             |             0.975 | [DCNN/DCNNDOY-1-sigmoid-conv12-he_uniform-l2norm-adam-labelSmoothing-fullset-classWeight.pkl](https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/DCNN/DCNNDOY-1-sigmoid-conv12-he_uniform-l2norm-adam-labelSmoothing-fullset-classWeight.pkl)                 |
| DCNN(w/o DOY)    |            0.9757 | [DCNN/DCNN-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight.pkl](https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/DCNN/DCNN-conv14-he_uniform-adam-labelSmoothing-fullset-classWeight.pkl)                                                         |
| ViT              |            0.9805 | [vision_transformer/ViT-default-20220509/predict_fullset.pkl](https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/vision_transformer/ViT-default-20220509/predict_fullset.pkl)                                                                                 |
| EfficientNet(B6) |            0.9782 | [efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224.pkl](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb6(transfer,trainable)-epochs150-batchsize64-lrreducer-sampleweight-224.pkl>) |
| EfficientNet(B3) |            0.9842 | [efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224.pkl](<https://github.com/Chang-Siang/farmland-crops/blob/main/src/evaluate/EfficientNet(fullset)/efficientnetb3(transfer,trainable)-epochs150-batchsize32-lrreducer-sampleweight-224.pkl>) |
