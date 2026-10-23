# 공간 네트워크 미로 복잡도

3차원 미로와 2차원 미로를 공간 네트워크 그래프로 읽어 아래 값을 계산한다.

- 트레모(Trémaux) 알고리즘 걸음 수
- 탐욕 라우팅 복잡도 `1 − 성공률`
- 기하평균 복잡도 `C_geo = √(τ_length × τ_degree)`
- 산술평균 복잡도 `C_arith = (τ_length + τ_degree) / 2`

`τ_length`는 간선 길이의 역수를 가중치로 둔 입구의 흡수시간이고, `τ_degree`는 도착 정점의 유효 차수 `max(2, deg)`를 쓴 흡수시간이다. 출구는 흡수 상태라 방정식에서 빼며, 연립방정식은 `np.linalg.solve`로 푼다. 축 가중치 네 가지(`standard`, `none`, `a`, `inv`)는 매번 모두 계산한다.

## 설치

Python 3.10 이상이 필요하다. 계산과 미로 생성은 numpy, 그림은 matplotlib와 plotly, 화면은 streamlit, 테스트는 pytest를 쓴다.

```text
pip install -r requirements.txt
```

## 입력 형식

좌표가 두 개 `[x, y]`이면 `z = 0`으로 채운다. 간선 길이는 두 끝의 택시거리다. `length`를 적었는데 택시거리와 다르면 오류다. 선택 항목 `meta`는 문자열과 숫자만 담은 객체다. 미로를 그래프로 바꿀 때도 그대로 넘어가고, CSV에는 `meta_<키>` 열이 된다.

### 그래프 형식

`examples/t3.json`과 같은 형식이다.

```json
{
  "nodes": {"A": [0, 0, 0], "B": [2, 0, 0], "C": [2, 3, 0]},
  "edges": [["A", "B"], ["B", "C"]],
  "entry": "A",
  "exit": "C"
}
```

### 미로 형식

파일에 `dim`이 있으면 미로로 보고, 곧은 통로는 정점에서 빼 간선 하나로 합친 뒤 계산한다. 정점 id는 `"x,y,z"`다. `dim`은 한 변의 칸 수이고, 칸 좌표는 `0` 이상 `dim − 1` 이하다. 격자 크기는 이 `dim` 값을 쓴다.

```json
{
  "dim": 3,
  "passages": [[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [2, 0, 0]]],
  "entry": [0, 0, 0],
  "exit": [2, 0, 0]
}
```

입구와 연결되지 않은 정점은 경고를 낸 뒤 제거하고, 이후 계산은 입구가 속한 연결 성분만 사용한다. 입구에서 출구로 가는 길이 없으면 오류다.

검산용 그래프는 `examples/t1.json`부터 `t5.json`, 그리고 정사각 고리 `examples/t8.json`에 있다. `examples/t6_*.json`은 일부러 틀린 입력이며, 읽으면 `ValueError`가 난다.

## 축 가중치

네 변형을 항상 함께 계산한다. `a(v)`는 정점 `v`에 닿은 간선이 쓰는 서로 다른 축의 개수다.

- `standard`: `w(v) = max(1, a(v) − 1)`
- `none`: `w(v) = 1`
- `a`: `w(v) = a(v)`
- `inv`: `w(v) = 1 / a(v)`

## 실행

```text
python -m mazecx.cli analyze examples/t3.json --runs 100 --seed 0
python -m mazecx.cli analyze examples/t3.json --runs 100 --seed 0 --out result.json
python -m mazecx.cli batch data/ --runs 100 --seed 0 --out results/results.csv
python -m mazecx.cli generate --out data/ --sizes 4 5 6 --loops 0 0.05 0.15 --count 20 --seed 0
python -m mazecx.cli generate --out data/ --sizes 4 5 --loops 0 0.15 --count 10 --seed 0 --dim2d
python -m mazecx.cli report results/results.csv --out site/
```

같은 파일과 같은 `--seed`면 트레모 결과(평균, 표준편차, 최소, 최대)가 다시 실행해도 같다. 반복은 난수 생성기 하나를 이어서 쓴다.

폴더 일괄 처리는 그 폴더 바로 아래의 `.json`만 파일명 순으로 읽는다. 하위 폴더는 들어가지 않는다. 파일 순번은 0부터이고, `i`번째 파일의 시드는 `seed + i`다. 첫 파일은 `analyze`에 같은 시드를 준 것과 같다. 파일 하나가 검증이나 계산에서 실패해도 멈추지 않고, 그 행의 `error` 열에 이유를 적은 뒤 다음 파일로 넘어간다. CSV는 UTF-8 BOM으로 저장한다.

`generate`는 각 `(크기 n, 고리 확률 p)`마다 `count`개의 미로를 만든다. `(0,0,0)`에서 무작위 깊이우선탐색으로 고리 없는 미로를 만든 뒤, 아직 벽인 이웃 칸을 확률 `p`로 뚫는다. 입구는 `(0,0,0)`, 출구는 `(n−1, n−1, n−1)`이다. `--dim2d`면 `n×n` 평면이고 출구는 `(n−1, n−1, 0)`이다. 파일 이름은 `maze_3d_n05_p0.050000_007.json`처럼 확률을 소수 여섯 자리로 고정해, 이름순으로 정렬해도 크기·확률·번호 순서가 유지된다. `meta`에는 생성기와 `n`, `loop_p`, 그 파일에 쓴 `seed`가 들어간다.

## 출력 항목

단일 실행의 JSON과 일괄 처리 CSV는 아래 순서를 쓴다. 그 뒤에 모든 파일의 `meta` 키 합집합을 사전순 `meta_<키>` 열로 붙이고, 없는 칸은 비운다.

```text
file, error, is_3d, grid_size, n_nodes, n_edges, n_loops, L_total, shortest_path,
tau_length_standard, tau_degree_standard, C_geo_standard, C_arith_standard,
tau_length_none, tau_degree_none, C_geo_none, C_arith_none,
tau_length_a, tau_degree_a, C_geo_a, C_arith_a,
tau_length_inv, tau_degree_inv, C_geo_inv, C_arith_inv,
tremaux_mean, tremaux_std, tremaux_min, tremaux_max, tremaux_norm,
greedy_success_rate, greedy_difficulty, greedy_entry_success, greedy_GR_score,
runs, seed, meta_*
```

- `error`: 성공하면 빈 문자열이고, 실패하면 이유 문구다.
- `is_3d`: 정점 `z` 좌표가 두 가지 이상이면 `true`.
- `grid_size`: 미로 형식이면 `dim`, 그래프 형식이면 축마다 `(최댓값 − 최솟값 + 1)` 가운데 최댓값.
- `n_loops`: 입구 연결 성분에서 `|E| − |V| + 1`.
- `shortest_path`: 입구에서 출구까지 간선 길이 기준 최단거리.
- `tremaux_std`: 걸음 수의 표본 표준편차(`n − 1`로 나눔). 반복이 1번이면 0.
- `tremaux_norm`: `tremaux_mean / L_total`.
- `greedy_difficulty`: `1 − greedy_success_rate`. 클수록 탐욕 라우팅이 어려운 미로다.
- `greedy_GR_score`: 출발점마다 성공이면 `최단거리 / 탐욕 경로 길이`, 실패면 0을 평균한 값.

축에 평행한 간선만 있는 그래프에서는 성공한 탐욕 경로가 최단 경로라 `greedy_GR_score`와 `greedy_success_rate`가 같다. 둘의 차이가 `1e-9` 이상이면 경고를 낸다.

## 보고서

`report`는 `error`가 비어 있는 행만 모아 스피어만 ρ, 95% 부트스트랩 구간, 피어슨 r을 계산하고 `site/index.html`과 SVG 그림 네 장을 만든다. 그림 안 글자는 영어이고, 설명은 HTML에 한국어로 적는다. 링크는 상대경로다.

웹 화면은 `app.py`다. 왼쪽에서 미로 하나 계산, 여러 미로 비교, 미로 만들기를 고른다.

여러 미로 비교는 탭 두 개다. 미로별 비교는 계산된 미로가 2개 이상일 때 쓸 수 있고, 트레모 평균 걸음 수, 탐욕 라우팅 복잡도, 우리 복잡도 기하평균, 우리 복잡도 산술평균을 표와 그래프로 보여 준다. 값이 클수록 어려운 미로이고, 화면 위의 축 변형을 바꾸면 표와 그래프가 함께 바뀐다. 전체 경향 분석은 성공한 미로가 8개 이상일 때 기존의 상관 분석을 보여 준다. 계산은 모두 계산을 눌렀을 때만 한다. 같은 파일 내용, 반복 횟수, 시드는 다시 계산하지 않고, 이름과 그룹은 표시용이다. 화면 2의 미로는 모두 같은 시드를 쓴다. 화면 1의 비교 목록에 추가를 누르면 방금 계산한 미로가 화면 2 목록에 들어간다.

## 로컬 실행

```text
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## 처음 배포하기

1. GitHub에 저장소를 만들고 이 프로젝트를 main 브랜치에 push한다.
2. https://share.streamlit.io 에 GitHub 계정으로 로그인한다.
3. Create app(또는 New app) → 저장소, 브랜치 main, Main file path에 app.py를 고른다.
4. App URL에 원하는 이름을 정하고, Advanced settings에서 Python 3.11을 고른 뒤 Deploy를 누른다.
5. 몇 분 뒤 https://<정한 이름>.streamlit.app 에서 프로그램이 열린다.
6. 이후 코드를 고쳐 main에 push하면 자동으로 새 코드로 바뀐다. push 전에 로컬에서 pytest가 통과하는지 꼭 확인한다(테스트가 실패해도 배포는 진행되기 때문).
