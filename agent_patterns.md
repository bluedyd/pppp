# ReAct: Synergizing Reasoning and Acting in Language Models

Yao et al., Princeton / Google Brain — ICLR 2023  
arxiv: https://arxiv.org/abs/2210.03629

## 핵심 아이디어

LLM의 **추론(Reasoning)** 과 **행동(Acting)** 을 분리해서 연구해왔던 기존 흐름을 합친다. Thought → Action → Observation을 번갈아 생성하여, 추론이 행동을 안내하고 행동(외부 환경)이 다시 추론을 보완하는 선순환 구조를 만드는 것.


## 기존 방식의 한계

| 방식 | 문제 |
|---|---|
| Chain-of-Thought (추론만) | 외부 정보 없이 내부 지식에만 의존 → **hallucination** 심각 |
| Act-Only (행동만) | 목표 분해, 상태 추적 불가 → 반복 행동, 루프에 빠짐 |



## ReAct 작동 방식

액션 스페이스를 `A ∪ L`로 확장. `L`은 언어 공간, 즉 **Thought**. Thought는 환경에 영향을 주지 않지만 컨텍스트를 업데이트해서 다음 행동을 더 잘 하게 만든다.

```
# https://github.com/ysymyth/ReAct
import json
import sys

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info
```

Thought의 역할은 다양함: 목표 분해, 부분 목표 추적, 상식 추론, 검색 쿼리 재구성, 예외 처리 등.



## 실험 결과

### 지식 집약적 태스크 (HotpotQA, FEVER)

- Wikipedia API (search / lookup / finish) 3가지 액션만 사용
- ReAct > Act 일관적으로 우세 (추론의 가치 입증)
- CoT보다 hallucination 비율 낮음 (false positive 6% vs 14%)
- **ReAct + CoT-SC 조합이 최고** — 내부 지식(CoT)과 외부 지식(ReAct) 상호 보완

### 인터랙티브 의사결정 태스크 (ALFWorld, WebShop)

- ALFWorld: ReAct **71%** vs Act-Only 45% vs BUTLER(IL, 10⁵ trajectories) 37%
- WebShop: ReAct **+10%** 절대 성능 향상 (IL+RL 대비)
- **단 1~2개 in-context 예시**만으로 대규모 학습 모델을 뛰어넘음

### 파인튜닝

- 프롬프팅만으로는 소형 모델(8B, 62B)에서 ReAct가 불리
- 3,000개 ReAct 궤적으로 파인튜닝하면 최고 성능 — 더 generalizable한 스킬을 학습하기 때문



## 의의 및 한계

### 의의

- 추론과 행동을 통합한 최초의 체계적 프레임워크
- 해석 가능성(interpretability)과 신뢰성 향상 — 사람이 Thought를 직접 편집해 에이전트 제어 가능
- Few-shot으로도 강력, 태스크 간 범용성 높음

### 한계

- Greedy decoding에서 반복 루프 취약
- 검색이 실패하면 회복이 어려움
- 대규모 프롬프팅 환경에서는 소형 모델 적용이 어려움 (파인튜닝 필요)

