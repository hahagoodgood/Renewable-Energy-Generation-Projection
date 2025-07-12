import pandas as pd
import numpy as np
# IterativeImputer가 아직 실험적인 기능이므로 활성화가 필요합니다.
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

def impute_precipitation_with_mice(input_file_path: str, output_file_path: str):
    
    """
    MICE(IterativeImputer)를 사용하여 ASOS 데이터의 강수 관련 컬럼 결측치를 대체합니다.

    이 함수는 다음 단계를 따릅니다:
    1. 데이터를 로드하고 '일시' 컬럼을 datetime 형식으로 변환합니다.
    2. 강수 관련 컬럼을 수치형과 시각(범주형)으로 분리합니다.
    3. 예측 성능 향상을 위해 다른 기상 변수들을 예측 변수로 선택합니다.
    4. MICE(IterativeImputer)를 사용하여 수치형 강수 컬럼의 결측치를 대체합니다.
    5. 시각 컬럼의 결측치는 각 컬럼의 최빈값(가장 자주 나타나는 값)으로 대체합니다.
    6. 대체된 값들이 물리적/논리적 일관성을 갖도록 후처리합니다.
    7. 최종 결과를 새로운 CSV 파일로 저장합니다.

    Args:
        input_file_path (str): 원본 데이터 CSV 파일 경로.
        output_file_path (str): 결측치 처리가 완료된 데이터를 저장할 CSV 파일 경로.
    """
    np.random.seed(42)
    print(f"'{input_file_path}' 파일 로딩 중...")
    # cp949 인코딩으로 원본 파일 로드
    try:
        df = pd.read_csv(input_file_path, encoding='cp949', low_memory=False)
    except UnicodeDecodeError:
        print("cp949 인코딩 실패. utf-8로 재시도합니다.")
        df = pd.read_csv(input_file_path, encoding='utf-8', low_memory=False)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {input_file_path}")
        return

    # --- 1. 데이터 전처리 및 컬럼 정의 ---
    # '일시' 컬럼을 날짜 형식으로 변환하여 월, 일 등 시간 특징 추출에 사용
    df['일시'] = pd.to_datetime(df['일시'])

    # 결측치를 채울 대상 컬럼 정의
    precipitation_cols = [
        '강수 계속시간(hr)', '10분 최다 강수량(mm)', '10분 최다강수량 시각(hhmi)',
        '1시간 최다강수량(mm)', '1시간 최다 강수량 시각(hhmi)', '일강수량(mm)'
    ]
    
    # MICE 적용을 위한 수치형 강수 컬럼
    numeric_precip_cols = [
        '일강수량(mm)', '강수 계속시간(hr)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)'
    ]
    
    # 최빈값으로 대체할 시각 컬럼
    time_precip_cols = [
        '10분 최다강수량 시각(hhmi)', '1시간 최다 강수량 시각(hhmi)'
    ]

    # MICE 모델의 예측 변수로 사용할 컬럼들 (예측 정확도 향상 목적)
    predictor_cols = [
        '평균기온(°C)', '최저기온(°C)', '최고기온(°C)',
        '평균 풍속(m/s)', '평균 상대습도(%)', '평균 현지기압(hPa)', '평균 해면기압(hPa)'
    ]
    
    # MICE에 사용할 전체 컬럼셋 (대상 컬럼 + 예측 변수)
    # mice_cols = numeric_precip_cols + predictor_cols
    mice_cols = precipitation_cols + predictor_cols

    print("결측치 대체 시작...")
    df_impute_subset = df[mice_cols].copy()

    # --- 2. MICE를 이용한 수치형 컬럼 대체 ---
    # IterativeImputer 초기화. RandomForestRegressor는 비선형 관계를 잘 포착하지만 계산 시간이 오래 걸릴 수 있습니다.
    # 더 빠른 실행을 원할 경우 estimator=BayesianRidge() 사용을 고려할 수 있습니다.
    # mice_imputer = IterativeImputer(
    #     estimator=RandomForestRegressor(n_estimators=5, random_state=42), # 빠른 실행을 위해 n_estimators 감소
    #     max_iter=5,          # 반복 횟수
    #     random_state=42,
    #     verbose=2            # 대체 과정 출력
    # )

    mice_imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=5, random_state=42), # 빠른 실행을 위해 n_estimators 감소 추후 200으로 수정
        max_iter=5,          # 반복 횟수 추후 10~20으로 수정
        random_state=42,
        verbose=2            # 대체 과정 출력
    )

    print("\n[MICE] 수치형 강수 데이터 결측치 대체 중...")
    imputed_data = mice_imputer.fit_transform(df_impute_subset)
    
    # 대체된 데이터를 다시 DataFrame으로 변환
    df_imputed_subset = pd.DataFrame(imputed_data, columns=mice_cols, index=df.index)

    # 원본 데이터프레임에 대체된 수치형 강수 컬럼 값을 업데이트
    for col in precipitation_cols:
        df[col] = df_imputed_subset[col]
        # 강수량 관련 값들은 음수가 될 수 없으므로 0으로 클리핑
        if '강수' in col or '시간' in col:
            df[col] = df[col].clip(lower=0)

    print("\n[MICE] 수치형 데이터 대체 완료.")

    # # --- 3. 최빈값을 이용한 시각 컬럼 대체 ---
    # print("[Mode] 시각 데이터 결측치 대체 중...")
    # for col in time_precip_cols:
    #     # 최빈값 계산 (결측치가 아닌 값들 중에서)
    #     mode_value = df[col].mode()
    #     df[col].fillna(mode_value, inplace=True)
    #     print(f" - '{col}' 컬럼의 결측치를 최빈값 '{mode_value}' (으)로 대체했습니다.")
    
    # --- 4. 논리적 일관성 후처리 ---
    print("\n[Post-processing] 논리적 일관성 검사 및 보정 중...")
    # 규칙 1: 일강수량이 0이면, 다른 모든 강수 관련 지표는 0이어야 함
    zero_rain_mask = df['일강수량(mm)'] <= 0.01 # 매우 작은 값도 0으로 간주
    for col in ['강수 계속시간(hr)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)']:
        df.loc[zero_rain_mask, col] = 0
    
    # 규칙 2: 상위 시간 단위의 강수량은 하위 시간 단위보다 크거나 같아야 함
    df['1시간 최다강수량(mm)'] = df[['일강수량(mm)', '1시간 최다강수량(mm)']].min(axis=1)
    df['10분 최다 강수량(mm)'] = df[['1시간 최다강수량(mm)', '10분 최다 강수량(mm)']].min(axis=1)
    
    
    print("논리적 보정 완료.")

    # --- 5. 결과 저장 ---
    print(f"\n결과를 '{output_file_path}' 파일로 저장 중...")
    df.to_csv(output_file_path, index=False, encoding='cp949')
    print("모든 작업이 성공적으로 완료되었습니다.")


# if __name__ == '__main__':
#     # 입력 파일 경로와 출력 파일 경로를 설정하세요.
#     INPUT_FILE = 'ASOS_통합(1301~2412)_fillna.csv'
#     OUTPUT_FILE = 'ASOS_통합_MICE_imputed.csv'
    
#     # 함수 실행
#     impute_precipitation_with_mice(INPUT_FILE, OUTPUT_FILE)