
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
* **CPU:** Intel(R) Core(TM) i7-14700KF(3.40 GHz)
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
| **Step 9** | Tiling + Padding (Bank Conflict Free) | 7.27s | -1.48 | 메모리 효율 증대, 입력 16비트 전환 제거 |
| **Step 10** | Kernel Fusion + Attention Optimized | 5.01s | -2.26 | 오버헤드 감소 |
| **Step 11** | Transposed Loading (Score Kernel) | 4.91s | -0.1 | 정확도 확보 및 가속 |
| **Step 12** | Parallel Reduction + Fast Math | 4.81s | -0.1 |  |
| **Step 13** | 배치 적용 | 4.64s | -0.17 | 최종 버전 |

> **Result:** 초기 대비 약 **0.0016배**, **-3091.36초**의 성능 향상을 달성하였으며, 결과값 검증(Comparator)을 통과하여 정확성을 입증했습니다.

<br>

## 🗂️ 프로젝트 핵심 파일 목록
* `Main.c`: 데이터 로드, OpenCL 환경 설정, 검증(Verification) 수행
* `ViT_opencl.c`: Host 코드. 버퍼 할당 및 커널 인자 설정, Enqueue 관리
* `kernel.cl`: **(핵심)** GPU에서 실행되는 OpenCL 커널 소스 코드
* `Network.h`: 가중치 로더 및 구조체 정의

<br>

## 📃 고찰 및 느낀 점 (Conclusion)

- 구동준 :

 본 프로젝트를 진행하면서 GPU 아키텍처의 메모리 계층 구조와 병렬 처리 모델에 대한 깊은 이해가 필요하다는 것을 깨달았습니다. 

 <br>

 코드를 병렬화하는 과정에서 다양한 문제들을 겪었습니다.
속도가 오히려 느려지는 경험도 많이 하였고, 빨라졌지만 결과값이 틀려지는 경험도 많이 하였습니다. 
커널 함수들을 전체적으로 수정할 때, 벡터화를 적용할 때 특히 많은 시행착오를 겪었으며, 문제가 생길 때마다 `git restore .`를 실행하며 이전 상태로 되돌리곤 했습니다.
이러한 수없이 많은 시행착오를 겪으며 문제 해결 능력이 향상되었고, 디버깅 스킬도 크게 발전했다고 생각합니다.

<br>
 
 마지막까지 해결되지 않은 문제도 있었습니다. 
호스트 코드의 배치화를 도전했습니다. 결과값이 틀리는 문제도 발생했고, 속도가 오히려 느려지는 문제도 발생했습니다.
최대한 시간을 투자해봤지만 이는 결국 해결하지 못했습니다.
 
 <br>

 이는 모두 소중한 경험이었고, 앞으로도 이러한 문제들을 해결해 나가며 성장해 나가야겠다는 다짐을 하게 되었습니다.


- 여승연 :
  
 순차 코드를 OpenCL 커널로 병렬 처리했을 때 실행 시간이 눈에 띄게 줄어드는 걸 보며 큰 성취감을 느꼈지만, 10초 이후부터는 성능이 잘 개선되지 않아 추가 최적화의 어려움도 실감할 수 있었습니다. 
 
 ViT를 사용해본 경험도 있고 관련 이야기를 많이 들어왔지만, 이론만 찾아봤지 코드적·하드웨어적 관점까지 깊게 파고들 생각은 하지 않았었습니다. 
 
 이번 프로젝트를 통해 하드웨어 수준의 프로그래밍을 직접 경험하며, 디버깅과 GPU 동작을 바라보는 시야가 한층 성장했음을 느낄 수 있었습니다.
