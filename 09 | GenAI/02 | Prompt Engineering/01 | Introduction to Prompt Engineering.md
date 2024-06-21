# Introduction to Prompt Engineering
#### 1. What is a Prompt?
- A **prompt** is an input to a Generative AI model, that is used to guide its output.
- Prompts may consists text, image, sound or any other media.
- **Examples:**
```
1. Write a three paragraph email for a marketing campaign for an accounting firm.
2. Describe everything on the table. Attachments cantains a table image.
3. Summarize the meeting. Attachments has a recording of an online meeting. 
```
- A **Prompt Template** is a function that contain one or more variables which will be replaced by some media to create a prompt.
- This prompt can be considered to be an instance of the template.
- **Examples:**
```
Template : Classify the tweet as positive or negative: {TWEET}
Prompt : Classify the tweet as positive or negative:
The NEET example paper leakage shows how this government has no regard for eductaion and development of this country. 
```
- Prompt and Prompt Template are distinct concepts a Prompt template becomes a prompt when we input someting in to the variables of the Prompt template. 

#### 2. Terminology 
##### 2.1. Components of a Prompt 
- Here are the most common components included in a prompt.

##### `Directive`
    
- Many prompts issue a directive in the form of an instruction or question.
- This is the core intent of the prompt, sometimes simply called the **"intent"**

- **Examples**
```
Tell me five books to read. 
```

- Directives can also be implicit, as in this one-shot case, where the directive is to perform English to spanish translation
```
Night: Noche
Morning:
```

##### `Examples`

- Examples are also known as exemplars or shots, act as demonstrations that guide GenAI to accomplish the task.
- The below above translation prompt is an One-Shot prompt.

##### `Output Formatting`

- It is often desirable for the GenAI to output information in certain formats, for example, CSVs or markdown format.
- **Example**

##### 2.2. Prompting Terms 
#### 3. A short History of Prompts 
