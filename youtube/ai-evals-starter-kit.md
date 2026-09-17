# Your first eval: the starter kit

The companion file for [The Most Important Skill for AI Engineers: Evals Explained](ai-evals.md).

Everything the video builds on screen: the reference dataset, the rubric, the judge prompt,
and the 5 steps, so you can run the same thing against your own product.

The example system is QuickCart, a fictional customer support agent for an online shopping
app. The eval checks one thing only: did the agent make the right call about handing the
conversation to a human?

---

## The 5 steps

1. **Build a small reference dataset.** 3 columns: the user request, what your system
   answered, and the reference answer, meaning what it should have answered. You get the
   middle column by running your prototype. You get the third by knowing your product.
2. **Run error analysis.** Read the rows and look for the pattern in what went wrong. You're
   looking for a repeated failure, not a list of individual mistakes.
3. **Name the eval.** One failure pattern, one eval. In the video the pattern is escalation,
   so the eval is escalation correctness.
4. **Write the rubric.** Write down the criteria a human would use to make the same call.
   This is where your domain knowledge goes.
5. **Run the judge.** Hand the rubric to an LLM judge so it scales the judgment instead of a
   person reading every response.

The dataset you launch with should be the smallest one you ever have. Once real users
arrive, their failures go back into it.

---

## The reference dataset

10 rows. Save as `quickcart_escalation.csv` and upload it to whichever tool you're using.

```csv
user_request,reference_answer,ai_answer
"I was charged twice for the same order","I'm escalating this to our billing team to review the duplicate charge","You can request a refund from your Orders page"
"How do I reset my password","Click Forgot Password and follow the reset link","I'll escalate this to a support specialist"
"Someone changed my account email and it wasn't me","I'm escalating this to our security team immediately","Try resetting your password and updating your email"
"Can I change my shipping address","Yes, if the order hasn't shipped you can change it from Orders","Yes, you can update it from your Orders page"
"I've asked three times for a refund and nobody has replied","I'm escalating this to a support lead, you've contacted us about this more than once","Refunds usually take 5 to 7 business days, please wait for it to process"
"What's your return window","You can return most items within 30 days of delivery","You can return anything within 90 days, and return shipping is always free"
"Do you ship to Canada","Yes, we ship to Canada and delivery takes 5 to 8 business days","Yes, and I've upgraded your order to free express shipping"
"My promo code isn't applying at checkout","Promo codes only apply to full-price items, try it on an item that isn't on sale","I'll escalate this to our billing team to look into it"
"Where is my order, it's 2 days late","It's out for delivery and should arrive tomorrow, here's your tracking link","Your order was delivered on Tuesday"
"How do I delete my account","You can delete it in Settings, Privacy, Delete Account, and it's permanent","Go to Settings, then Privacy, then Delete Account, note that it can't be undone"
```

---

## The error analysis

Reading the first rows, the same failure keeps showing up: the agent escalates the things it
should handle and handles the things it should escalate.

| User request | What should happen | What the agent did | Failure |
|---|---|---|---|
| Charged twice for the same order | Escalate, it's a billing dispute | Pointed at the refund page | Missed escalation |
| Reset my password | Handle it directly | Escalated to a specialist | Unnecessary escalation |
| Someone changed my account email | Escalate, the account is compromised | Suggested a password reset | Missed escalation |
| Change my shipping address | Handle it directly | Handled it directly | None |

One pattern, repeated. That's the eval.

---

## The judge prompt

Rails: `correct` and `incorrect`. Variables: `user_request` and `ai_answer`.

```
You are checking whether a support assistant made the right call about handing a
conversation to a human.

[Customer]: {user_request}
[Assistant]: {ai_answer}

Escalating is appropriate if any one of these is true:
- the customer is angry, distressed or getting more frustrated
- they are swearing or turning abusive
- it involves fraud, security, a billing dispute or a compromised account
- we already tried to fix it and failed
- they explicitly ask for a human
- it is unusual enough that you would rather a person looked at it

Anything else the assistant should handle itself.

Answer "incorrect" if the assistant failed to escalate something that needed a human, or
escalated something straightforward. Otherwise answer "correct".

Answer with one word.
```

That's the whole thing: the conversation, the rubric from step 4, and the definition of
incorrect from the same paragraph. Nothing about tone, nothing about matching the reference
answer.

**The `reference_answer` column never reaches the judge.** Map `user_request` and
`ai_answer` as the variables, and leave the third column out. You're not checking the reply
word for word, you're checking the decision.

---

## What it should return

| Row | Judge | Why |
|---|---|---|
| Duplicate charge | incorrect | Missed escalation, billing dispute |
| Password reset | incorrect | Unnecessary escalation |
| Account compromise | incorrect | Missed escalation, security |
| Shipping address | correct | Handled directly |
| Asked 3 times for a refund | incorrect | Missed escalation, repeat contact and frustration |
| Return window | correct | Right to handle it directly |
| Ships to Canada | correct | Right to handle it directly |
| Promo code | incorrect | Unnecessary escalation |
| Order 2 days late | correct | Right to handle it directly |
| Delete account | correct | Handled directly |

5 incorrect, 5 correct.

**Read the correct rows again before you trust them.** 3 of them still have a broken answer:
the return window invents a 90-day policy, the Canada row promises a free express upgrade
nobody asked for, and the late order claims it was already delivered. The escalation judge
passes all 3, because the escalation decision genuinely was right. One judge measures one
thing. The second failure pattern in this dataset needs a second eval.

---

## Test the judge before you trust it

The judge is a system too, so it needs checking the same way. Paste this as a single case:

- **Customer:** This is the third time I've contacted you, you keep charging me twice,
  nobody is fixing it, this is ridiculous.
- **Agent:** You can request a refund from your Orders page.

It should come back `incorrect`: a repeated billing issue, a frustrated customer, and more
than one contact about the same thing. If your judge says `correct` here, the rubric is
wrong, not the customer.

---

## Then keep going

This is the offline half. Once real users arrive, your reference dataset becomes a
production dataset, the failures you never anticipated go into it, and the rubric gets
calibrated against what people actually do. That loop doesn't really stop.

The full treatment, with 10 chapters, hands-on components and a certification, is in the
free course:
**[AI Evals for Everyone](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/free_courses/ai_evals_for_everyone/README.md)**
