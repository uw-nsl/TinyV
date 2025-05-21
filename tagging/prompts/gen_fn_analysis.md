## Task Description

You are an AI tasked with identifying false negatives in answer verification. A false negative occurs when a model's answer is essentially correct but is marked as incorrect due to minor discrepancies or formatting issues. Your job is to analyze the given question, ground truth answer, and model answer to determine if the model's answer is actually correct despite appearing different from the ground truth.

Analyze the inputs carefully, considering the following:
1. Is the model's answer mathematically equivalent to the ground truth?
2. Are there minor formatting differences that don’t affect the answer’s correctness?
3. Is the model's answer more precise or in a different but valid format?

## Examples

Here are some examples of questions, ground truth answers, and model answers. All of them are correct.

**Example 1 (Order-Insensitive):**
<question>Determine all real values of $x$ for which $(x+8)^{4}=(2 x+16)^{2}$.</question>
<ground_truth_answer>-6,-8,-10</ground_truth_answer>
<model_answer>-10, -8, -6</model_answer>

<analysis>
```json
{
  "reasoning": "The model's answer lists the same values as the ground truth but in a different order. Since the question asks for all solutions, the order doesn't matter for correctness.",
  "is_correct": true
}
```
</analysis>

**Example 2 (Latex Expression):**
<question>A bag contains 3 green balls, 4 red balls, and no other balls. Victor removes balls randomly from the bag, one at a time, and places them on a table. Each ball in the bag is equally likely to be chosen each time that he removes a ball. He stops removing balls when there are two balls of the same colour on the table. What is the probability that, when he stops, there is at least 1 red ball and at least 1 green ball on the table?</question>
<ground_truth_answer>$\\frac{4}{7}$</ground_truth_answer>
<model_answer>4/7</model_answer>

<analysis>
```json
{
  "reasoning": "The model's answer '4/7' is mathematically equivalent to the ground truth answer '$\\frac{4}{7}$'. The only difference is in the notation - the ground truth uses LaTeX fraction notation while the model uses a simple division format. The numerical value is identical in both cases."
  "is_correct": true
}
```
</analysis>

**Example 3 (Variable):**
<question>If $T=x^{2}+\\frac{1}{x^{2}}$, determine the values of $b$ and $c$ so that $x^{6}+\\frac{1}{x^{6}}=T^{3}+b T+c$ for all non-zero real numbers $x$.</question>
<ground_truth_answer>-3,0</ground_truth_answer>
<model_answer>b=-3, c=0</model_answer>

<analysis>
```json
{
  "reasoning": "The model's answer 'b=-3, c=0' is mathematically equivalent to the ground truth answer '-3,0'. The model simply labeled the values with their corresponding variables, which provides more clarity but doesn't change the mathematical content of the answer.",
  "is_correct": true
}
```
</analysis>

**Example 4 (Paraphrase):**
<question>Peter has 8 coins, of which he knows that 7 are genuine and weigh the same, while one is fake and differs in weight, though he does not know whether it is heavier or lighter. Peter has access to a balance scale, which shows which side is heavier but not by how much. For each weighing, Peter must pay Vasya one of his coins before the weighing. If Peter pays with a genuine coin, Vasya will provide an accurate result; if a fake coin is used, Vasya will provide a random result. Peter wants to determine 5 genuine coins and ensure that none of these genuine coins are given to Vasya. Can Peter guaranteedly achieve this?</question>
<ground_truth_answer>Petya can guarantee finding 5 genuine coins.</ground_truth_answer>
<model_answer>Yes, Peter can guarantee finding 5 genuine coins while ensuring that none of these genuine coins are paid to Vasya.</model_answer>

<analysis>
```json
{
  "reasoning": "The model's answer correctly states that Peter can guarantee finding 5 genuine coins, which matches the ground truth. The model provides additional details about ensuring none of these coins are paid to Vasya, but this doesn't change the correctness of the answer."
  "is_correct": true
}
```
</analysis>

## Input

Now, please analyze the following question, ground truth answer, and model answer.

<question>
{{QUESTION}}
</question>

<ground_truth_answer>
{{GROUND_TRUTH_ANSWER}}
</ground_truth_answer>

<model_answer>
{{MODEL_ANSWER}}
</model_answer>

## Output

Please provide your analysis in the following JSON format:
<analysis>
```json
{
  "reasoning": "Your detailed reasoning here",
  "is_correct": true/false
}
```
</analysis>

Ensure your reasoning is thorough and considers all aspects of the answers. The "is_correct" field should be true if the model's answer is essentially correct despite any minor differences from the ground truth and false otherwise.