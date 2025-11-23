# Cognitive Tasks Token Limit Fix

## Problem
The `cognitive_tasks_battery` was often failing because token limits were too low for well-thought-out responses:
- Math problems: 150 tokens (insufficient for step-by-step reasoning)
- Instruction tasks: 100 tokens (too restrictive)
- Summarization: 100 tokens (not enough for comprehensive summaries)
- Creative tasks: 150 tokens (too short for detailed creative responses)

## Solution

### 1. Increased Token Limits
Updated default token limits based on task complexity:

- **Math/Logic Problems**: 150 → **300 tokens**
  - Allows for step-by-step reasoning and explanations
  
- **Instruction Adherence**: 100 → **200 tokens**
  - Gives enough space to follow complex instructions properly
  
- **Summarization Tasks**: 100 → **Auto-calculated** (min 250, typically `target_length * 4 + 50`)
  - Scales with target word count
  - Example: 25-word target → ~150 tokens minimum
  
- **Creative Tasks**: 150 → **300-400 tokens**
  - Stories: 400 tokens (need more elaboration)
  - Other creative tasks: 300 tokens

### 2. Token Limit Information in Prompts
Added token limit information to prompts so models can self-regulate:

```python
# Before:
"Solve this problem step by step: {problem.problem_text}"

# After:
"Solve this problem step by step. You have approximately {max_tokens} tokens to complete your response. Show your work: {problem.problem_text}"
```

This helps models:
- Plan their response length
- Avoid running out of tokens mid-thought
- Complete tasks within the allocated budget

### 3. Configurable Token Limits
Made token limits configurable via constructor parameters:

```python
test = CognitiveTasksTest(
    model_name="gpt2",
    test_mode=True,
    max_tokens_math=300,              # Math problems
    max_tokens_instruction=200,       # Instruction tasks
    max_tokens_summarization=None,    # Auto-calculated if None
    max_tokens_creative=None          # Auto-calculated if None
)
```

### Implementation Details

#### Math Problems
```python
max_tokens = self.max_tokens_math  # Default: 300
prompt = f"Solve this problem step by step. You have approximately {max_tokens} tokens to complete your response. Show your work: {problem.problem_text}"
```

#### Instruction Tasks
```python
max_tokens = self.max_tokens_instruction  # Default: 200
prompt = f"{task.instruction_text} (You have approximately {max_tokens} tokens to complete this task.)"
```

#### Summarization Tasks
```python
# Auto-calculated: ~4 tokens per word + buffer
if self.max_tokens_summarization is None:
    max_tokens = max(250, task.target_length * 4 + 50)
else:
    max_tokens = self.max_tokens_summarization
prompt = f"Summarize the following text in approximately {task.target_length} words. You have approximately {max_tokens} tokens to complete your response. Be concise but comprehensive: {task.source_text}"
```

#### Creative Tasks
```python
# Stories get more tokens
if self.max_tokens_creative is None:
    max_tokens = 400 if task.task_type == "story" else 300
else:
    max_tokens = self.max_tokens_creative
prompt = f"{task.prompt} (You have approximately {max_tokens} tokens to complete your response. Be creative and detailed.)"
```

## Benefits

1. **Fewer Failures**: Higher token limits reduce truncation issues
2. **Better Quality**: Models can complete thoughts and provide full answers
3. **Self-Regulation**: Models know their token budget and can plan accordingly
4. **Flexibility**: Token limits can be adjusted per task type or model
5. **Scalability**: Summarization limits scale with target length

## Files Modified

- `neuromod/testing/cognitive_tasks.py`:
  - Increased default token limits
  - Added token limit information to prompts
  - Made limits configurable via constructor
  - Added auto-calculation for summarization tasks

## Usage

### Default (Recommended)
```python
test = CognitiveTasksTest(model_name="gpt2", test_mode=True)
# Uses optimized defaults: 300 math, 200 instruction, auto-calc for others
```

### Custom Limits
```python
test = CognitiveTasksTest(
    model_name="gpt2",
    test_mode=True,
    max_tokens_math=500,           # More tokens for complex math
    max_tokens_instruction=300,   # More for complex instructions
    max_tokens_summarization=400,  # Fixed limit for summaries
    max_tokens_creative=500        # More for creative tasks
)
```

## Impact

- **Reduced Failures**: Tasks should complete successfully more often
- **Better Responses**: Models can provide complete, well-reasoned answers
- **Improved Evaluation**: More complete responses lead to better assessment
- **Flexibility**: Can adjust limits based on model capabilities or requirements

