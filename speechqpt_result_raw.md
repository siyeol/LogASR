# 실험 결과 정리

> **TL;DR.** per-tensor activation INT4 커널이 실제로 작동하는 제약 내에서, ASR 모델들의 speech encoder의 heavy-tail 때문에 uniform A4가 ASR을 붕괴시킨다. 따라서 그 분포에 맞는 log codebook을 사용하면 W4A4 PTQ가 산다는 것을 입증한다.

## 1. 배경 및 기존 연구의 한계

### a. LLM PTQ의 기본 가정

- Weight는 group/GPTQ 등의 기법으로 살리고, activation outlier는 드물고 한/두 채널에 존재함.
- 따라서 SmoothQuant로 채널을 옮기거나 token/group scale로 잘라내면 activation 양자화 가능.

### b. 하지만 speech encoder는

1. 입력이 단어 한 개가 아니라 긴 시간축 $\times$ 넓은 FFN.
2. 이상치가 한/두 채널에만 존재하는 것이 아니라, 대부분의 채널이 heavy-tail을 가짐.
3. 그래서 SmoothQuant가 도움이 되지 않고, GPTQ/AWQ도 weight-only quantization이기 때문에 크게 도움되지 않음.

> **Figure (PDF p.1): Activation channel outliers -- LibriSpeech test-clean ($n=2620$).**  
> Whisper encoder, Qwen encoder, Qwen decoder의 channel-rank curves를 비교하며, speech encoder 쪽에서 outlier tail이 더 넓게 분포하는 현상을 보여준다.

### c. 왜 Activation이 제약인가?

1. Weight group ($g=64$)는 scale을 한 번 저장하면 되지만, activation group은 매 forward마다 동적 absmax이기 때문에 사용되지 않음.
2. Baseline (W4A8): weight -- uniform, $g=64$ / activation -- uniform, $g=0$ (per-tensor) / SQ ($\alpha=0.5$).
3. 제안 기법 (W4A4): Activation을 log codebook으로 변경.

### SmoothQuant (베이스라인과 제안 공통, 코드북 아님)

채널 $j=1,\ldots,K$, $\alpha=0.5$:

$$
s_j=rac{\max_i |X_{i,j}|^{\alpha}}
{\max_i |W_{j,i}|^{1-\alpha}},
$$

$$
X\leftarrow X\,\operatorname{diag}(s)^{-1},
\qquad
W\leftarrow W\,\operatorname{diag}(s).
$$

그다음 W는 2, A는 3을 한다.

### 2. Grouped uniform INT4 weight (공통, $g=64$)

$K$를 길이 $g=64$ 그룹으로 나눔. 출력 $n$, 그룹 $t$에 대해

$$
A^w_{n,t}=\|W_{n,\,tg:tg+g}\|_{\infty},
\qquad
\Delta^w_{n,t}=\frac{A^w_{n,t}}{7},
$$

$$
y=\operatorname{clip}\!\left(
\operatorname{round}\!\left(\frac{W_{n,\,tg:tg+g}}{\Delta^w_{n,t}}\right),-7,7
\right),
$$

$$
\widehat{W}_{n,\,tg:tg+g}=y\,\Delta^w_{n,t}.
$$

Weight는 log가 아니다. Scale도 row-wise가 아니라 row, group이다.

### 3. 제안: grouping-free log activation (여기가 새 수식)

토큰 $t$마다 last-dim absmax (구현 `_absmax(x, dim=-1)`):

$$
A_t=\|X_{t,:}\|_{\infty},
\qquad
\alpha_t=\frac{A_t}{127}.
$$

이렇게 두면

$$
\log_2\!\left(1+\frac{A_t}{\alpha_t}\right)=7
$$

이라 최댓값이 격자 끝 $y=7$에 붙는다.

$$
f(x)=\operatorname{sign}(x)\log_2\!\left(1+\frac{|x|}{\alpha_t}\right),
$$

$$
y=\operatorname{clip}\!\left(\operatorname{round}(f(x)),-7,7\right),
$$

$$
\widehat{x}=\operatorname{sign}(y)\,\alpha_t\left(2^{|y|}-1\right).
$$

Activation 그룹 없음 ($g_a=0$). Scale은 토큰당 하나뿐이다.

### 4. 대조 베이스라인: grouping-free uniform A (표의 붕괴 행)

같은 W-SQ를 사용하고 A만 다름. 텐서 전체 absmax (`uniform_dyn`):

$$
A=\|X\|_{\infty},
\qquad
\Delta=\frac{A}{7},
$$

$$
y=\operatorname{clip}\!\left(\operatorname{round}(x/\Delta),-7,7\right),
\qquad
\widehat{x}=y\Delta.
$$

여기를 이미지처럼 같은 $A$, 같은 per-token으로 쓰면 안 된다. 실험은 **per-tensor uniform vs per-token log**이다.

### d. Log-codebook의 역할

1. 4-bit 격자 15칸 $(-7,\ldots,7)$에 실수 축을 균등 배치하는 경우, 몸통은 거칠고 꼬리가 잘림 $\rightarrow$ encoder FFN이 깨지고 디코더는 EOS/hallucination error 발생.
2. $\mu$-law 꼴의 log-scale은 거의 선형으로, 큰 값은 접는다 (mel-scale like) $\rightarrow$ 15칸을 몸통에 더 쓰고도 꼬리를 범위 안에 넣음.
3. 꼬리: $|x|$가 absmax에 가까운 소수 / 몸통: 나머지 대부분. Whisper Encoder FFN은 median이 약 $0.015\,\text{absmax}$.

## 2. 왜 A8이 아니라 A4인가?

1. Encoder FFN activation은 매 layer마다 $T\times D$로, 여기를 8 bit로 두면 4 bit의 두 배.
2. GEMM input byte 감소.
3. 커널: log index는 4-bit로 저장 $\rightarrow$ INT8 GEMM.

## 3. 실험 결과

### a. Ablation study

| Ablation list | Whisper-large-v3 test-clean WER (%) | Whisper-large-v3 test-other WER (%) | Qwen3-ASR-1.7B test-clean WER (%) | Qwen3-ASR-1.7B test-other WER (%) |
|---|---:|---:|---:|---:|
| FP16 | 1.92 | 3.91 | 1.64 | 3.41 |
| Encoder-only W4A4 | 2.16 | 4.92 | 2.22 | 5.43 |
| Attention-only W4A4 | 2.12 | 4.35 | 1.95 | 4.34 |
| FFN-only W4A4 | 2.12 | 4.72 | 2.06 | 4.57 |
| Decoder-only W4A4 | 2.54 | 4.87 | 94.26 | 94.67 |
| Weight-only log-scale W4A4 | 2.13 | 4.33 | 1.85 | 4.26 |
| log-scale W4A4 (W, A 전체) | 2.76 | 6.47 | 2.96 | 8.12 |
| **Ours (Activation-only log-scale W4A4)** | **2.62** | **6.16** | **2.22** | **5.43** |

### b. Comparison with existing method

| Methods | Whisper-large-v3 test-clean WER (%) | Whisper-large-v3 test-other WER (%) | Qwen3-ASR-1.7B test-clean WER (%) | Qwen3-ASR-1.7B test-other WER (%) |
|---|---:|---:|---:|---:|
| FP16 | 1.92 | 3.91 | 1.64 | 3.41 |
| RTN (W4A8) | 1.95 | 4.05 | 1.80 | 3.73 |
| AWQ + per-tensor RTN (W4A8) | 2.03 | 4.36 | 1.82 | 3.86 |
| GPTQ + per-tensor RTN (W4A8) | 1.93 | 4.09 | 1.73 | 3.83 |
| SmoothQuant + per-tensor RTN (W4A8) | 2.02 | 4.07 | 1.88 | 4.04 |
| Ours (W4A8) | 1.92 | 3.93 | 1.72 | 3.76 |
| SmoothQuant + per-tensor RTN (W4A4) | 99.71 | 99.55 | 104.71 | 104.78 |
| **Ours (W4A4)** | **2.62** | **6.16** | **2.22** | **5.43** |

### c. Efficiency metric comparison

| Model | Precision | WER (LS test-clean, %) | WER (LS test-other, %) | H2D memory (MiB) | GEMM input byte | Throughput (utt/s @4GB) | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| Whisper-large-v3 | FP16 | 1.92 | 3.91 | 2944 | 5747 | 17.0 | 0.013 |
| Whisper-large-v3 | Baseline (W4A8) | 2.02 | 4.07 | 843 | 1950 | 18.9 | 0.017 |
| Whisper-large-v3 | Ours (W4A4) | 2.62 | 6.16 | 843 | 1627 | 18.9 | 0.017 |
| Qwen3-ASR-1.7B | FP16 | 1.64 | 3.41 | 3887 | 636.4 | 13.7 | 0.017 |
| Qwen3-ASR-1.7B | Baseline (W4A8) | 1.88 | 4.04 | 1428 | 170.4 | 41.9 | 0.025 |
| Qwen3-ASR-1.7B | Ours (W4A4) | 2.22 | 5.43 | 1428 | 159.1 | 41.9 | 0.020 |

### d. Domain generalization

| Model | Precision | AMI | Earnings22 | VoxPopuli | GigaSpeech | CommonVoice | LS-c REVE (SNR RT60 1.2]s |
|---|---|---:|---:|---:|---:|---:|---:|
| Whisper-large-v3 | FP16 | 16.27 | 11.36 | 9.29 | 10.10 | 9.86 | 6.25 |
| Whisper-large-v3 | Ours (W4A4) | 18.86 | 13.13 | 10.65 | 10.88 | 15.79 | 15.26 |
| Qwen3-ASR-1.7B | FP16 | 10.25 | 10.24 | 6.37 | 8.73 | 7.13 | 3.85 |
| Qwen3-ASR-1.7B | Ours (W4A4) | 14.06 | 12.04 | 7.48 | 9.60 | 11.74 | 10.08 |

### e. Silence packaging

1. **문제:** 실제 음성 입력에는 중복되거나 무의미한 silence frame이 상당수 포함되어 있어, 인코더-디코더 전반의 토큰 연산량을 불필요하게 낭비함.
2. **제안 기법:** Silence Frame Packaging: 연속적인 묵음 구간을 단일 토큰 형태로 압축 패키징하여 음향 문맥 손실 없이 인코더 시퀀스 길이를 대폭 감축.
3. **실험 결과:** Sequence length는 Whisper의 경우 frame 수, Qwen3-ASR의 경우 token 수.

| Model | Method | LibriSpeech test-clean WER | LibriSpeech sequence length | AMI WER | AMI sequence length |
|---|---|---:|---:|---:|---:|
| Whisper-large-v3 | Baseline (FP16) | 1.92 | 370.5 frames | 16.21 | 1342.2 frames |
| Whisper-large-v3 | Ours (FP16) | 1.95 | 338.7 frames | 19.91 | 960.1 frames |
| Whisper-large-v3 | Ours (W4A4) | 3.12 | 338.7 frames | 20.56 | 960.1 frames |
| Qwen3-ASR-1.7B | Baseline (FP16) | 1.64 | 96.7 tokens | 10.25 | 349.2 tokens |
| Qwen3-ASR-1.7B | Ours (FP16) | 1.65 | 88.1 tokens | 11.46 | 249.6 tokens |
| Qwen3-ASR-1.7B | Ours (W4A4) | 2.27 | 88.1 tokens | 15.80 | 249.6 tokens |

## FAQ

- **log-scale의 novelty?** grouping-free A4에서 ASR이 죽는 현상을 보임과 그 분포를 매칭해서 W4A4 PTQ 성공한 것이 contribution.
- **A8을 사용하면 되지 않나?** ASR 모델에서 W4A4의 최초 연구.
- **GPTQ/AWQ + A4?** SQ에서 동일하게 비교함.
