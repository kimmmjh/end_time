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
