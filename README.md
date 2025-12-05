# Vision Transformer (ViT) OpenCL Acceleration Project

<br>

## 📋 프로젝트 개요 (Project Overview)
본 프로젝트는 **Vision Transformer (ViT-B/16)** 모델의 추론(Inference) 과정을 **OpenCL**을 사용하여 GPU 가속화하는 것을 목표로 합니다. 순차적으로 실행되는 CPU 기반의 코드를 분석하고, 병렬 처리(Parallelism)와 메모리 계층 구조(Memory Hierarchy)를 고려한 최적화 커널을 작성하여 성능을 극대화하였습니다.

* **목표:** ViT 모델의 End-to-End 추론 속도 가속
* **제약 사항:** 
    * 기존 딥러닝 알고리즘 구조 변경 불가
    * CPU 실행 결과와 **100% 일치**하는 정확도 유지 (Verification Pass 필수)

<br>

## 💻 개발 환경 (Environment)
* **CPU:** ( 확인중 )
* **GPU:** NVIDIA GeForce GTX 4060
* **OS:** Windows 11
* **Framework:** OpenCL 
* **Compiler:** Visual Studio (MSVC)

<br>

## 📈 핵심 최적화 기법 (Key Optimization Techniques)
본 프로젝트에서는 GPU 하드웨어 특성을 고려하여 단계별 최적화를 수행했습니다.

### 1️⃣ Memory Optimization
1. **Vectorized Memory Access (float4):** 
    
    `Conv2d` 및 `Linear` 커널에서 `float4` 자료형을 사용하여 메모리 대역폭 효율을 4배 극대화하고, 인스트럭션 수를 감소시켰습니다.

2. **Local Memory Tiling:** 
    
    행렬 곱(GEMM) 연산 시, 자주 사용되는 데이터를 온칩 메모리인 **Local Memory(Shared Memory)**에 적재하여 Global Memory 접근을 최소화했습니다.

3. **Bank Conflict 제거:** 
    
    Tiling 과정에서 `[16][16]` 배열 대신 **`[16][17]` Padding**을 적용하여 Shared Memory의 Bank Conflict를 제거, 접근 지연을 방지했습니다.

4. **Transposed Loading:** 
    
    Attention Score ($Q \times K^T$) 계산 시, $K$ 행렬을 로딩 단계에서 즉시 전치(Transpose)하여 저장함으로써 메모리 접근 패턴을 최적화했습니다.

### 2️⃣ Compute Optimization

1. **Parallel Reduction:** 

    `LayerNorm`과 `Softmax` 연산에 **Work-Group 단위의 병렬 리덕션(Tree-based Reduction)** 알고리즘을 적용하여, 단일 스레드 처리 대비 GPU 코어 활용률(Occupancy)을 극대화했습니다.

2. **Loop Unrolling:** 

    반복문 내의 연산을 펼쳐 분기 예측 비용을 줄이고 명령어 파이프라인 효율을 높였습니다.

3. **Fast Math:** 

    정확도 손실이 미미한 범위 내에서 `native_exp`, `native_rsqrt` 등 하드웨어 가속 초월 함수를 사용하여 연산 속도를 향상했습니다.

### 3️⃣ System Optimization
* **Stream Pipelining:** 

    OpenCL Command Queue를 활용하여 Host-to-Device 데이터 전송과 커널 실행을 비동기적으로 중첩(Overlap)시켜 전체 지연 시간을 은폐(Hiding)했습니다.

* **Kernel Fusion:** 

    `Linear + GELU`, `Linear + Add(Residual)` 연산을 하나의 커널로 융합하여 Global Memory I/O 비용과 커널 실행 오버헤드를 제거했습니다.

<br>

## 📑 성능 평가 (Performance Evaluation)
총 100장의 이미지에 대한 평균 추론 시간을 측정하였습니다.

| 단계 | 최적화 내용 | 실행 시간 (ms) | Speedup | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | ViT 모델 순차적 실행 시간 | 3096s | 1.0x | 기준점 |
| **Step 1** | 미리 할당한 버퍼 2개 재사용  | 3056.36s | -39.74s |  |
| **Step 2** | Conv2d, Linear, Attention, Softmax 연산들을 커널 함수로 변환 |  |  | 
| **Step 3** | Step 2의 함수들에 대해 로컬 메모리 타일링 적용 | 10.7s | -3045.66s | 
| **Step 4** | 오버래핑 적용 | 10.72s | 0s |  |
| **Step 5** | 입력 16bit 전환 | 10.03s | -0.69s | 조건 위반으로 인하여 step 9부터 적용 취소 |
| **Step 6** | unwrap 적용 | 9.66s | -0.37s |  |
| **Step 7** | 커널 함수 인자 일부에 `__global` 대신 `__constant` 적용 | 9.55s | -0.11s |  |
| **Step 8** | 컴파일러 옵션에 Fast Math 적용 | 8.75s | -0.8s |  |
| **Step 9** | Tiling + Padding (Bank Conflict Free) | (re-Conv2d_MLP 실행 시간) | | 메모리 효율 증대, 입력 16비트 전환 제거 |
| **Step 10** | Kernel Fusion + Attention Optimized | (re-MHA 실행 시간) |  | 오버헤드 감소 |
| **Step 11** | Transposed Loading (Score Kernel) | (re-LN_Softmax 실행 시간) |  | 정확도 확보 및 가속 |
| **Step 12** | Parallel Reduction + Fast Math | (re-Native 실행 시간) |  |  |

> **Result:** 초기 대비 약 **n배**, **-n초**의 성능 향상을 달성하였으며, 결과값 검증(Comparator)을 통과하여 정확성을 입증했습니다.

<br>

## 🗂️ 프로젝트 핵심 파일 목록
* `Main.c`: 데이터 로드, OpenCL 환경 설정, 검증(Verification) 수행
* `ViT_opencl.c`: Host 코드. 버퍼 할당 및 커널 인자 설정, Enqueue 관리
* `kernel.cl`: **(핵심)** GPU에서 실행되는 OpenCL 커널 소스 코드
* `Network.h`: 가중치 로더 및 구조체 정의

<br>

## 📃 결론 및 고찰 (Conclusion)

작성 중 : 

예시 : 

본 프로젝트를 통해 단순히 알고리즘을 GPU로 옮기는 것만으로는 성능 향상에 한계가 있음을 깨달았습니다. GPU 아키텍처의 **Memory Hierarchy(Global vs Local)**와 **Parallelism(Thread vs Work-Group)**을 깊이 이해하고, **Bank Conflict**나 **Memory Coalescing** 같은 하드웨어 병목을 해결했을 때 비로소 유의미한 가속을 이룰 수 있었습니다. 특히 디버깅이 어려운 병렬 프로그래밍 환경에서 정확도를 유지하며 최적화하는 경험을 통해 GPGPU 프로그래밍에 대한 깊은 이해를 얻었습니다.
