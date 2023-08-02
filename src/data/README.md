# Data Description

## label.csv

- [label.csv](label.csv) 為官方提供的原始資料標註檔，訓練階段使用，共 80,270 筆
- 主辦方於 2022/04/15 宣布將 `inundated` 類別合併至`bareland` 類別，進行實驗的時候請留意

  | 欄位名稱 | 欄位說明 |
  | -------- | -------- |
  | filename | 檔案名稱 |
  | category | 作物類別 |

## describe.csv

- [describe_train_fullset.csv](describe_train_fullset.csv) 在官方提供資料 [label.csv](label.csv) 再加入影像描述資料，共 80,270 筆，並預先以 85%、15% 的比例切分為訓練（Training）與驗證（Validation）資料集
- [describe_train_subset.csv](describe_train_subset.csv) 為 [describe_train_fullset.csv](describe_train_fullset.csv) 之子集合（25%），依據類別與拍攝月份等比例取樣，共 20,065 筆，並預先以 70%、15%、15% 的比例切分為訓練（Training）與驗證（Validation）與測試（Testing）資料集
- [describe_train_fullset_valid_test.csv](describe_train_fullset_valid_test.csv) 從 [describe_train_fullset.csv](describe_train_fullset.csv) 的 Validation 再切出 20% 當作 Test 資料，用於衡量集成模型的績效
- [describe_test.csv](describe_test.csv) 為官方測試影像的影像描述資料，不包含真實類別，共 20,000 筆

  | 欄位名稱              | 欄位說明                                   |
  | --------------------- | ------------------------------------------ |
  | path                  | 檔案名稱                                   |
  | label                 | 作物類別                                   |
  | county_name           | 縣市名稱                                   |
  | town_name             | 鄉鎮名稱                                   |
  | make                  | 拍攝相機的製造商                           |
  | model                 | 拍攝相機的型號                             |
  | taken_datetime        | 影像拍攝的時間                             |
  | day_of_year           | 日期參數，表示一年中的第幾天               |
  | transform_day_of_year | 日期參數，表示一年中的第幾天，頭尾連續版本 |
  | set_name              | 資料屬於 train、valid 或 test              |
