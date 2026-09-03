# BB Neural BP message updates

현재 BB decoder는 각 Tanner edge에서 `I, X, Y, Z` 네 상태의 log-message를
유지한다. 한 BP iteration은 다음 순서로 진행된다.

```text
V -> C message
    -> exact BP4 C -> V update
    -> neural residual + learned relaxation
    -> posterior와 다음 V -> C message
```

기본 설정에서는 이 과정을 12번 반복한다. 12는 measurement round가 아니라
BP algorithm iteration 수이며, 모든 iteration이 같은 neural-network
parameter를 공유한다.

## BP iteration

BP iteration은 Tanner graph의 모든 edge message를 한 번 갱신하는 단계다.
초기 V -> C message는 noise prior이고, 초기 C -> V message는 uniform
distribution이다. 이후 한 iteration에서 다음 세 단계를 수행한다.

```text
1. 모든 check에서 exact C -> V message 계산
2. C -> V에 neural residual과 relaxation 적용
3. 모든 qubit의 posterior와 다음 V -> C message 계산
```

이를 기본 12번 반복한다.

```text
prior로 message 초기화
    -> iteration 1
    -> iteration 2
    -> ...
    -> iteration 12
    -> 최종 qubit posterior
```

한 mini-batch의 12개 iteration이 모두 끝난 뒤 loss를 계산하고 한 번
backpropagation한다. BP iteration마다 optimizer step을 수행하는 것은 아니다.

## 기호

- `v`: physical-qubit variable node
- `c`: stabilizer check node
- `P`: Pauli 상태, `P in {I, X, Y, Z}`
- `N(v)`, `N(c)`: 해당 node의 Tanner-graph 이웃
- `pi_v(P)`: noise model에서 얻은 qubit `v`의 Pauli prior
- `s_c`: 측정된 check syndrome bit
- `m_{v->c}(P)`, `m_{c->v}(P)`: 정규화된 log-message

## V -> C update

Variable node는 channel prior와 목적지 `c`를 제외한 다른 check message를
더한다.

$$
m_{v\to c}(P)
=
\operatorname{logsoftmax}_{P}
\left[
\log \pi_v(P)
+\sum_{c'\in N(v)\setminus\{c\}}m_{c'\to v}(P)
\right].
$$

이 부분은 일반적인 BP4 update를 그대로 사용하며 신경망을 사용하지 않는다.

## Exact C -> V update

Check node는 주변 Pauli들이 측정 syndrome parity를 만족할 확률을 계산한다.
`a_c(P)`를 Pauli `P`가 check `c`와 anticommute하면 1, commute하면 0이라고
하자.

```text
X check와 anticommute: Y, Z
Z check와 anticommute: X, Y
```

이웃 `u`의 message로부터 anticommute 확률 `q_{u->c}`를 계산하면, 후보
`P`에 대한 syndrome-compatible 확률은 다음과 같다.

$$
\widetilde p_{c\to v}(P)
=
\frac{1}{2}
\left[
1+(-1)^{s_c+a_c(P)}
\prod_{u\in N(c)\setminus\{v\}}
\left(1-2q_{u\to c}\right)
\right].
$$

Exact BP4 message는 이 확률을 log-space에서 정규화한 값이다.

$$
m^{\mathrm{exact}}_{c\to v}(P)
=
\operatorname{logsoftmax}_{P}
\left[\log \widetilde p_{c\to v}(P)\right].
$$

## Neural residual

신경망은 exact C -> V message를 대체하지 않고 작은 residual을 더한다.
각 edge의 MLP 입력은 총 13차원이다.

```text
exact C -> V message       4
현재 V -> C message        4
이전 C -> V message        4
해당 check syndrome bit    1
----------------------------
합계                       13
```

MLP 구조는 다음과 같다.

```text
LayerNorm(13)
    -> Linear(13, 64)
    -> SiLU
    -> Linear(64, 4)
    -> tanh * 2
```

이 구조를 사용한 이유는 다음과 같다.

- `exact C -> V`, `현재 V -> C`, `이전 C -> V`를 함께 보면 vanilla BP가
  제안한 현재 값과 message 변화 방향을 모두 참고할 수 있다.
- syndrome bit는 해당 check가 원하는 even/odd parity를 알려준다.
- `LayerNorm`은 log-message scale 차이를 줄여 학습을 안정화한다.
- hidden width 64는 edge마다 실행하기에 작으면서 비선형 correction을 표현할
  수 있는 기본값이다.
- 출력 4개는 `I, X, Y, Z` message에 각각 더할 residual이다.
- `tanh * 2`는 residual을 제한해 신경망이 exact BP message를 한 번에 지나치게
  크게 파괴하는 것을 막는다.
- 마지막 Linear를 0으로 초기화하므로 학습 전에는 residual이 정확히 0이다.

따라서 neural submodule의 출력은 edge별 Pauli log-message residual이다.

```text
residual shape: [batch, number_of_edges, 4]
```

Residual과 함께 edge orbit별 relaxation coefficient도 학습한다.

$$
\lambda=1+0.5\tanh(r), \qquad 0.5<\lambda<1.5.
$$

최종 C -> V message는 다음과 같다.

$$
m^{\mathrm{new}}_{c\to v}
=
\operatorname{logsoftmax}
\left[
\lambda m^{\mathrm{exact}}_{c\to v}
+(1-\lambda)m^{\mathrm{old}}_{c\to v}
+R_{c\to v}
\right].
$$

이 neural correction은 12개 BP iteration 모두에서 적용된다. 같은 edge
orbit은 위치와 iteration에 관계없이 같은 MLP를 사용하지만, 입력 message가
달라지므로 residual 값은 매번 달라진다.

## Equivariance가 반영되는 부분

Equivariance는 pooling이나 MLP 내부의 특별한 layer에서 생기는 것이 아니라,
BB Tanner edge의 **orbit별 parameter sharing**에서 생긴다.

BB code를 cyclic translation하면 check, qubit, edge 위치는 바뀌지만 edge가
속한 polynomial orbit은 변하지 않는다. 현재 구현은 12개 orbit을 사용한다.

```text
X/Z check
    x left/right qubit block
    x polynomial displacement 3개
    = 12 edge orbits
```

같은 orbit의 모든 translated edge가 같은 residual MLP와 같은 relaxation
coefficient를 사용한다. Exact BP의 sum/product aggregation도 edge permutation에
무관하므로, syndrome을 cyclic shift하면 출력 qubit posterior도 같은 방식으로
shift된다.

$$
f(T_gs)=T_gf(s).
$$

여기서 `T_g`는 BB cell의 cyclic translation이다. 12개 iteration에서 같은
MLP를 재사용하는 것은 **iteration sharing**이고, translated edge에서 같은
MLP를 사용하는 것이 **spatial equivariance**다. 두 개념은 서로 다르다.

## Loss

MLP가 출력하는 residual에는 별도의 정답이 없다. 즉, 특정 residual 값을
맞히는 MSE loss를 사용하지 않는다. Residual을 적용한 전체 BP decoder의
qubit posterior에 loss를 걸고, gradient가 12개 BP iteration을 거슬러 MLP로
전달된다.

각 iteration 출력에 사용하는 기본 loss는 다음과 같다.

$$
L_t
=
L_{\mathrm{syndrome}}
+L_{\mathrm{logical}}
+0.1L_{\mathrm{Pauli}}.
$$

- `Syndrome loss`: 예측 correction의 syndrome이 입력 syndrome과 같아지도록
  하는 binary cross entropy
- `Logical loss`: 실제 error와 예측 correction의 residual이 trivial logical
  coset에 속하도록 하는 loss
- `Pauli loss`: 각 qubit의 실제 `I, X, Y, Z`를 맞히는 보조 cross entropy

마지막 12번째 출력의 loss가 주 loss이고, 앞의 11개 iteration에도 deep
supervision을 적용한다.

$$
L_{\mathrm{total}}
=
L_{12}
+0.2\frac{1}{11}\sum_{t=1}^{11}L_t.
$$

따라서 MLP는 residual 자체를 직접 학습하는 것이 아니라, 최종 physical
correction이 syndrome을 만족하고 logical error를 남기지 않도록 residual을
간접적으로 학습한다.

## 초기화와 최종 출력

Residual MLP의 마지막 layer는 0으로, relaxation은 `lambda=1`로
초기화한다. 따라서 학습 전에는 Neural BP가 vanilla BP4와 정확히 같다.

12번째 iteration이 끝나면 각 qubit의 posterior를 출력한다.

$$
\ell_v(P)=
\operatorname{logsoftmax}_{P}
\left[
\log\pi_v(P)+\sum_{c\in N(v)}m_{c\to v}(P)
\right].
$$

```text
BB72 output:  [batch, 72, 4]
BB144 output: [batch, 144, 4]
```

마지막 차원의 순서는 `I, X, Y, Z`이다. 각 qubit에서 `argmax`를 취한
`[batch, n]` Pauli 배열이 hard correction이 된다. Residual은 작은 MLP의
중간 출력이고, 전체 decoder의 최종 출력은 qubit별 Pauli posterior이다.

---

# Circuit-level BB Neural BP2

위 설명은 code-capacity BP4이고, circuit-level decoder는 Stim의 detector
error model(DEM) 위에서 **binary BP2**를 수행한다.

```text
여러 round의 noisy Stim circuit
    -> DEM: fault probability + detector/observable signature
    -> detector--fault-mechanism Tanner graph
    -> normalized min-sum BP2 + neural residual
    -> 각 fault mechanism의 posterior LLR
```

- detector 하나가 check node가 된다.
- DEM fault mechanism 하나가 binary variable node가 된다. 같은 detector와
  observable signature를 가진 항들은 graph를 만들 때 합쳐질 수 있다.
- 해당 mechanism이 detector를 뒤집으면 두 node 사이에 edge를 만든다.
- measurement round와 fault propagation은 이미 DEM 안에 들어 있다.
- `T=12`는 measurement round 수가 아니라 이 고정 graph에서 반복하는 BP
  iteration 수다.

최종 출력 shape은 `[batch, num_mechanisms]`이다. 원소는 각 mechanism이
발생하지 않았다는 쪽의 posterior log-likelihood ratio(LLR)이며, 양수일수록
그 mechanism이 발생하지 않았을 가능성이 높다.

## Code-level과 달라진 점

Circuit-level 모델은 code-level Tanner graph를 DEM graph로 단순 교체한
것만은 아니다. 추론하는 variable과 message의 의미도 바뀐다.

| | Code-level BP4 | Circuit-level BP2 |
|---|---|---|
| variable node | physical data qubit | DEM fault mechanism |
| check node | stabilizer | detector |
| graph | `Hx`, `Hz` | Stim DEM에서 만든 detector--mechanism graph |
| variable 상태 | `I, X, Y, Z` | 발생하지 않음/발생함 |
| BP message | 4-state log probability | scalar binary LLR |
| baseline update | sum-product BP4 | normalized min-sum BP2 |
| neural output | C-to-V residual 4개 | C-to-V scalar residual 1개 |
| 최종 posterior | `[B, n, 4]` | `[B, num_mechanisms]` |
| 시간 정보 | 없음 | 모든 measurement round가 DEM graph에 포함 |

Code-level에서는 `(check type, spatial edge orbit)`마다 별도 residual MLP를
사용한다. Circuit-level DEM은 훨씬 크고 irregular하므로, MLP 본체 하나를
모든 edge가 공유하고 space-time orbit embedding과 orbit별 relaxation으로
edge의 역할을 구분한다. 기본값은 `sharing=orbit`이며 `global`이 아니다.

## Physical error와 DEM fault mechanism

Circuit noise instruction 하나는 gate, reset, measurement 또는 idle 위치에서
Pauli fault를 발생시킬 수 있다. Stim은 그 fault가 회로를 따라 전파되었을 때
뒤집는 detector와 logical observable을 계산해 다음과 같은 DEM mechanism으로
표현한다.

```text
probability p_j
    + detector signature {D2, D7, ...}
    + observable signature {L0, ...}
```

모델은 회로 전체에 fault가 하나라도 있는지를 하나의 binary 값으로 분류하지
않는다. DEM에 있는 각 mechanism `j`에 대해 posterior LLR `ell_j`를 낸다.

```text
ell_j > 0: mechanism j가 발생하지 않았을 가능성이 큼
ell_j < 0: mechanism j가 발생했을 가능성이 큼
```

DEM mechanism은 microscopic physical fault와 항상 1:1은 아니다. 서로 다른
gate fault라도 detector와 observable signature가 같으면 graph를 만들 때 같은
effective mechanism으로 합칠 수 있다. 반대로 한 mechanism은 여러 detector를
동시에 뒤집을 수 있다.

따라서 decoder의 목표는 실제 microscopic fault history를 정확히 복원하는
것이 아니다. 같은 syndrome을 설명하는 fault set은 여러 개일 수 있으므로,

```text
관측 detector parity를 만족하고
    +
실제 error와 합친 뒤 logical error를 남기지 않는
correction-equivalent mechanism set을 찾는 것
```

이 목표다. 학습 때 Stim이 알려주는 sampled mechanism label은 보조
`mechanism BCE`에만 사용한다. 주된 `detector loss`와 `logical loss`는
degeneracy를 허용하므로, 예측 mechanism set이 실제 sampled set과 달라도
올바른 logical correction이면 성공으로 학습할 수 있다. 평가 때는 mechanism
label을 decoder에 제공하지 않는다.

## 한 circuit-level BP iteration

먼저 신경망을 넣지 않은 normalized min-sum baseline의
detector-to-mechanism message `m_exact`를 계산한다. 그 다음 learned
relaxation과 residual을 적용한다. 여기서 `exact`는 코드에서 neural
correction 전의 값을 부르는 이름이며, 전체 확률추론이 exact라는 뜻은 아니다.

$$
m^{new}_{c\to v}
=
\lambda_g m^{exact}_{c\to v}
+(1-\lambda_g)m^{old}_{c\to v}
+R_\theta(x_{cv}, e_g).
$$

여기서 `g`는 edge sharing group이고, 기본값 `orbit`에서는 같은
space-time orbit의 edge가 같은 embedding과 relaxation을 공유한다.
MLP 본체 하나는 모든 orbit이 공유하고, orbit별 차이는 embedding과
relaxation으로 준다.

- `lambda_g = 1 + 0.5 tanh(a_g)`, 따라서 기본 설정에서 `0.5 < lambda_g < 1.5`
- `R_theta = 2 tanh(MLP(...))`, 따라서 residual은 `[-2, 2]`
- 갱신한 message는 마지막에 `[-30, 30]`으로 clip한다.

그 뒤 mechanism posterior를 계산하고, 목적지 edge의 message를 제외한
extrinsic mechanism-to-detector message를 다음 iteration으로 보낸다.

## Residual MLP의 입력과 구조

각 Tanner edge와 각 shot마다 다음 8개 scalar feature를 만든다.

| 번호 | edge feature | 역할 |
|---:|---|---|
| 1 | 현재 exact C-to-V message | vanilla min-sum이 제안한 값 |
| 2 | 현재 V-to-C message | mechanism 쪽에서 들어온 정보 |
| 3 | 이전 C-to-V message | 직전 iteration의 상태 |
| 4 | 해당 detector bit | 현재 shot에서 detector가 click했는지 |
| 5 | mechanism prior LLR | noise model이 주는 사전 확률 |
| 6 | mechanism posterior LLR | 현재까지 모인 주변 정보 |
| 7 | 정규화한 detector degree | irregular DEM graph의 local degree |
| 8 | V-to-C message의 절댓값 | 현재 message의 confidence 크기 |

기본 `orbit_embedding_dim=8`에서는 여기에 learned orbit embedding 8개를
붙인다.

```text
8 edge features + 8-dimensional orbit embedding
                    -> 16
                    -> LayerNorm
                    -> Linear(16, 32)
                    -> SiLU
                    -> Linear(32, 1)
                    -> tanh x 2
                    -> scalar residual
```

이 MLP가 작은 이유는 이 함수를 모든 shot, 모든 Tanner edge, 모든 BP
iteration에서 실행하기 때문이다. 계산량과 activation memory가 대략
`batch x edge 수 x BP iteration 수`에 비례하므로 큰 network는 빠르게
비싸진다. 또한 mechanism은 발생/비발생의 binary variable이어서 BP
message 자체가 scalar LLR이다. 따라서 residual도 scalar 하나면 충분하다.

- `LayerNorm`: syndrome bit, degree, prior LLR처럼 scale이 다른 feature를
  안정적으로 섞는다.
- `Linear(16, 32)`: local BP state 사이의 비선형 interaction을 표현하되
  parameter와 memory를 작게 유지한다.
- `SiLU`: 12번 unroll하여 미분할 때 부드러운 activation을 제공한다.
- 마지막 `Linear(32, 1)`: 한 C-to-V LLR에 더할 scalar correction을 만든다.
- 마지막 layer의 zero initialization: 학습 시작 시 residual이 정확히 0이고
  `lambda=1`이므로 vanilla normalized min-sum BP와 동일하다.

`8`, `8`, `32`, `SiLU`, `LayerNorm` 자체는 이론적으로 유일한 정답이 아니라
현재의 안정성/비용 trade-off를 위한 hyperparameter다.

## 이 MLP가 만족해야 하는 조건

현재 설계에서 중요한 조건은 다음과 같다.

1. **출력 의미**: binary C-to-V message에 더할 유한한 scalar LLR
   correction이어야 한다. Class probability나 logical class 출력이 아니다.
2. **Equivariance**: `sharing=orbit`일 때 absolute detector/edge ID가 아니라
   symmetry로 대응되는 edge orbit이 같은 parameter를 써야 한다. Orbit
   assignment, shared MLP, permutation-invariant BP aggregation이 함께 이
   성질을 만든다. Pooling이 equivariance를 만드는 것은 아니다.
3. **Local/graph-size-independent update**: edge 하나의 local state를 같은
   MLP로 처리하므로 BB72, BB144 또는 다른 round 수에서도 MLP 입출력 shape은
   같다. 다만 graph buffer와 orbit 수에 따른 embedding table은 새 graph에
   맞춰 다시 만들며, orbit 수가 다른 checkpoint는 그대로 load할 수 없다.
4. **안정적인 unrolling**: residual과 relaxation이 bounded이고 전체 message가
   finite해야 한다. 현재 `tanh`, relaxation 범위, message clipping이 이를
   담당한다.
5. **Vanilla BP fallback**: residual이 0이고 `lambda=1`이면 정확히 baseline
   normalized min-sum으로 돌아가야 한다. 그래야 같은 shot에서 neural BP와
   vanilla BP를 공정하게 비교할 수 있다.
6. **정보 누출 방지**: 입력은 현재 shot의 detector/BP state와 고정된 graph
   metadata만 사용해야 하며, 실제 발생 mechanism label이나 logical 정답을
   forward input으로 넣으면 안 된다.

MLP 자체가 parity constraint를 직접 만족할 필요는 없다. Parity 구조는 DEM
Tanner graph와 baseline min-sum update가 제공하고, MLP는 그 update에 bounded
local correction만 더한다. 또한 orbit embedding만 붙인다고 자동으로
equivariant가 되는 것은 아니며, symmetry에 맞는 orbit 분류가 전제되어야
한다.

## Circuit-level loss

Residual의 정답값을 따로 주고 MSE로 학습하지 않는다. 최종 mechanism
posterior에 다음 end-to-end loss를 적용한다.

$$
L
=
L_{detector}
+L_{logical}
+0.1L_{mechanism}
+0.2L_{deep}.
$$

- `detector loss`: 예측 mechanism parity가 관측 detector bit를 재현하게 한다.
- `logical loss`: 실제 fault와 예측 correction을 합친 residual의 logical
  observable parity가 trivial이 되게 한다.
- `mechanism loss`: Stim이 sampling한 mechanism label에 대한 작은 보조 BCE다.
  Degeneracy 때문에 이것을 주 loss로 사용하지 않는다.
- `deep supervision`: 마지막을 제외한 중간 BP iteration loss의 평균이다.

## Stim data와 noise model

Circuit은 perfect reference cycle, 지정한 수의 noisy syndrome-extraction
cycle, perfect closing cycle로 구성된다. 따라서 `rounds=R`이면 detector
frame은 `R+1`개다. Stim은 회로의 모든 noise instruction과 measurement
propagation을 분석해 DEM을 만들고, 이 repo는 그 DEM을 Tanner graph로
변환한다.

`--bb_circuit_noise_model`로 다음 profile을 선택한다.

| profile | 주요 channel |
|---|---|
| `legacy` | reset flip `p`, H 뒤 `DEP1(p)`, CNOT 뒤 `DEP2(p)`, configurable measurement `q`, optional data idle |
| `standard` | preparation/measurement `p`, ideal H, CNOT `DEP2(p)`, 매 physical tick의 inactive qubit에 `DEP1(p)` |
| `si1000` | reset `2p`, H와 gate idle `p/10`, CNOT `p`, measurement `5p`, M/R tick의 resonator idle `2p` |

`standard`와 `si1000`은 arXiv:2607.05897의 Table II/III channel rate를 현재
periodic BB seven-CNOT-layer schedule에 적용한다. 논문의 open-boundary layout과
routing 자체를 그대로 복제한 것은 아니다. SI1000의 여러 idle channel은
서로 독립적인 Stim instruction으로 쌓이며, 표에 있는 SWAP `1.5p`는 현재
schedule에 SWAP이 없으므로 실행되지 않는다.

학습에서는 DEM sampler의 `return_errors=True`를 사용해 detector,
observable, sampled mechanism label을 얻는다. 평가는 exact Stim circuit
sampler로 새 detector/observable shot을 만들며 mechanism label은 사용하지
않는다. 즉, 학습과 평가는 같은 물리 noise profile을 쓰지만 평가가 latent
fault 정답을 decoder input으로 보는 일은 없다.
