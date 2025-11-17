Morris1337dank
morris1337dank
I ett samtal

ylva — 2025-11-15 22:13
Vidarebefordrat
Mohamad Changes:
Added 3 habit frequency sensors (daily/weekly/monthly) with full habit lists
Created daily motivation sensor with personalized messages
Implemented habitica_unscored_task_alert event for automation triggers
Added dynamic polling based on user activity

My Changes after that:
Created sensor entity class for motivation (HabiticaMotivationalSensor)
Built complete dashboard example with all new sensors
Configured icons for all new entities
Created automation examples for the alert event

It was a little bit tough, first Mohamad changes were not working at all on my environment and he had to fix it (almost 2 hours to fix it, I think this due to our different environments), but once this was fixed my implementation was easy to build on top of Mohamad’s.
Vidarebefordrat
Changes are now live!
Vidarebefordrat
Let me know if you need enything else from me/us
detta har han skickat
kollar på det imorn!
kan ha möte närsom imorn om du vill ska plugga hela dan
Morris1337dank — Igår 12:57
Aha nice! Märkligt att han inte skickade till mig
ylva — Igår 12:57
ja han kke tänkte att jag skulle skicka till dig
men det sa han inget om lol
Morris1337dank — Igår 12:59
Jag kan typ inte förrän 18 tyvärr, men ska oavsett jobba på det ikväll!
ylva — Igår 13:00
ok lugnt! jag sitter o försöker förstå NLP uppgiften fortfarande, är mer orolig över den haha
fast lär gå o lösa helt ok fort med gpt
jag kan börja skriva och strukturera upp det vi fått från joao så kan vi se vad som e kvar ikväll o dela upp det!
Morris1337dank — Igår 16:26
är redo redan nu faktiskt, går in i SEP-voice chat
ylva — Igår 16:43
source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
ylva — Igår 17:02
Okay so that I have understood correctly, you have implemented everything except FR1 that allows you to add a new habit? In that case we will leave it for assignment 4.  And the sensor entity class, is that part of the modularization that we talked about?
jobsgonnawork — 3:25 PM
Exactly, that adding functionality will come after!
ylva — 3:26 PM
ok toppp
then what im gonna comment is basically about this
Image
jobsgonnawork — 3:27 PM
The new sensor entity classes were designed thinking of modularisation, we just need to extract them, create new files and a better folder structure I would say
jobsgonnawork — 3:28 PM
That’s good
I think you have everything there
ylva — 3:29 PM
Ok thats good! I can make up some stuff and reason according to our discussion the other day, but then from this table im wondering 1: did you create any tests? And I guess the modularization solves the first point, that parts are clearly distinguished. The second though?
jobsgonnawork — 3:35 PM
Yes you do that! 
1: I did not create specific tests (pass/fail) but I created 2 YML examples that we can use for testing, to check if everything is the way it should be
2: yes modularisation solves the first one, didn’t catch that “second thought”
ylva — 3:35 PM
okay okay i can write that
hahah i "the second though" I was reffering to the API Volatility, just making sure that the API logic is encapsulated?
jobsgonnawork — 3:42 PM
ahahah sorry, still a little sleepy i guess, yes all the API logic is inside coordinator.py
ylva — 4:12 PM
And then I dont remember fully but you guys were supposed to create the uml diagram and stuff for what you have done? im reffering to this:  Diagrams Create one or more design sketches (compo-
nent, class, or sequence diagrams) showing how your new code
interacts with the existing system. Design sketches can be
lightweight—simple UML or box-and-arrow diagrams are suffi-
cient—as long as they clearly show structure and dependencies.
I think we talked to mohammed at least about this, have you guys talked about it?
ylva — Igår 20:25
ok skrev lite o la till lite ordbajs också, feel free att ta bort eller ändra
typ bara mina ord o ingen gpt iaf! på gott o ont  haha
Morris1337dank — Igår 20:26
Nice! Jag ska sätta mig snart och kika på det
Morris1337dank — Igår 21:55
Jag känner mig färdig nu tror jag, jag tyckte det såg bra ut och ändrade bara någon enda mening för flow.  Känns som att vi resonerat kring allt, förhoppningsvis håller gruppen med haha
ylva — Igår 22:46
ok nice 👍👍👍
Morris1337dank — Igår 23:00
👍 , har du gjort några ändringar i NLP förresten? Jag har versionen du pushade för 11h sen
ylva — Igår 23:01
asså bara tränat en till modell men inte ändrat nåt i koden! gjorde ett script för att kunna återanvända det tokenizade datasetet men det visade sig gå snabbt jämfört med träningsloopen så va typ ingen poäng med det
ylva — 10:30
tjo
är du igång o jobbar eller? såg att du pushat lite grejer igår, har du gjort nåt mer sen dess?
Morris1337dank — 10:34
Jag sitter med det men har inte skrivit någon kod än, förstår mig inte helt på MHA:n😅
ylva — 10:36
eeh nä inte hag heller
lol
ok ringer om 10?
ylva
 startade ett samtal som varade i 38 minuter. — 10:55
ylva
 startade ett samtal som varade i 26 minuter. — 14:04
ylva — 14:13
Bild
Morris1337dank — 15:52
Bild
Du missade ett samtal från 
ylva
 som varade i några sekunder. — 16:53
ylva — 16:54
e tbks o jobbar nu
hmm jaaa har stött på det där punkt tab grejen också
men det va ett problem o sen tror jag inte jag fattade hur det försvann? men tror man kan lägga till en download punkt tab ba
Morris1337dank
 startade ett samtal som varade i 40 minuter. — 17:01
ylva — 17:39
hahahaha
glömde att jag va connectad
Morris1337dank — 17:39
hahahahahahaha
hörde precis ljudet
ylva — 17:39
joinar strax
ylva
 startade ett samtal som varade i 2 timmar. — 17:58
ylva — 20:53
ok är tillbaka!
Morris1337dank — 20:54
den e färdig
ylva
 startade ett samtal. — 20:54
Morris1337dank — 21:23
Bild
Morris1337dank — 22:49
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.distributions import Categorical
Expandera
message.txt
9 KB
ylva — 23:40
import torch
import numpy as np
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

local_dir = "/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"
Expandera
eval_pre.py
6 KB
Loaded model on: cuda

Example text: The grass is green, the sky is blue, the apple is

Top-5 next word predictions:
1.  is           (prob: 0.8243)
Expandera
evalresultsPRE.txt
8 KB
﻿
import torch
import numpy as np
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

local_dir = "/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"

tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)

###
### Load model
###
model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loaded model on:", device)

# Determine a pad token ID (HF tokenizer may not have one)
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0


###
### Sampling helper
###
def sample_text(model, tokenizer, prompt, max_length=50, temperature=1.0, topk=0, device=None):
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    enc = tokenizer([prompt], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
    input_ids = enc.clone()

    eos_id = tokenizer.eos_token_id
    prompt_len = input_ids.size(1)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # (1, T, V)

        last_logits = logits[0, -1, :].float()  # (V,)

        if temperature == 0:  # greedy
            next_id = int(torch.argmax(last_logits).cpu().item())
        else:
            scaled = last_logits / float(max(1e-8, temperature))

            if topk and topk > 0:
                topk_vals, topk_idx = torch.topk(scaled, k=topk)
                distr = Categorical(logits=topk_vals)
                sel = int(distr.sample().cpu().item())
                next_id = int(topk_idx[sel].cpu().item())
            else:
                distr = Categorical(logits=scaled)
                next_id = int(distr.sample().cpu().item())

        # Append new token
        next_tensor = torch.tensor([[next_id]], device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

    # Prepare output
    all_ids = input_ids[0].tolist()
    prompt_ids = all_ids[:prompt_len]
    gen_ids = all_ids[prompt_len:]

    def ids_to_tokens(ids):
        toks = []
        for tid in ids:
            if tid == pad_id:
                continue
            if tokenizer.bos_token_id is not None and tid == tokenizer.bos_token_id:
                continue
            if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
                break
            toks.append(tokenizer.convert_ids_to_tokens(int(tid)))
        return toks

    prompt_tokens = ids_to_tokens(prompt_ids)
    gen_tokens = ids_to_tokens(gen_ids)
    return prompt_tokens, gen_tokens, all_ids


###
### Detokenizer for nicer printing
###
def detokenize(tokens):
    no_space_before = {",", ".", "?", "!", ";", ":", "%", ")", "]", "}"}
    contractions = {"n't", "'s", "'re", "'ve", "'ll", "'d"}
    no_space_after = {"(", "[", "{"}

    out = ""
    for tok in tokens:
        if out == "":
            out = tok
        elif tok in no_space_before or tok in contractions:
            out += tok
        elif out and out[-1] in no_space_after:
            out += tok
        else:
            out += " " + tok
    return out


###
### Next-word prediction example
###
example_text = "The grass is green, the sky is blue, the apple is"
print("\nExample text:", example_text)

enc = tokenizer([example_text], truncation=True, padding=True, return_tensors="pt")["input_ids"].to(device)
input_ids = enc[0]

# Find last non-pad token
last_pos = max(i for i, tok in enumerate(input_ids.tolist()) if tok != pad_id)

# Run model only on non-pad prefix
outputs = model(enc[:, :last_pos])
logits = outputs.logits[0, -1]  # (V,)
topk = torch.topk(logits, k=5)
indices = topk.indices.tolist()
probs = torch.softmax(topk.values, dim=0).tolist()

print("\nTop-5 next word predictions:")
for rank, (idx, prob) in enumerate(zip(indices, probs), 1):
    word = tokenizer.decode([idx], skip_special_tokens=True)
    print(f"{rank}. {word:12s}  (prob: {prob:.4f})")


###
### Sampling examples
###
examples = [
    "In natural language processing, a Transformer",
    "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
    "Write a Python program that reverses a list.",
    "To be or not to be, that is",
    "Oscar Piastri is a",
    "A good career choice for graduate software engineers is",
]

print("\n--- Sampling examples ---")
for prompt in examples:
    print(f"\nPrompt: {prompt}")
    for temp, k in [(0.0, 0), (0.7, 10), (0.7, 40), (1.0, 60)]:

        prompt_toks, gen_toks, all_ids = sample_text(
            model, tokenizer, prompt,
            max_length=50, temperature=temp, topk=k, device=device
        )

        # Decode the entire generated sequence (prompt + continuation)
        decoded_full = tokenizer.decode(all_ids, skip_special_tokens=True)

        # Decode the prompt alone (clean, readable text)
        decoded_prompt = tokenizer.decode(
            tokenizer.encode(prompt, add_special_tokens=False),
            skip_special_tokens=True
        )

        # Extract only the generated continuation
        continuation = decoded_full[len(decoded_prompt):].strip()

        print(f" temp={temp:0.2f}, topk={k} | prompt: {decoded_prompt}")
        print(f"               continuation: {continuation[:300]}")