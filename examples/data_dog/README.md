---
competition_id: financial-well-being-sme
platform: zindi
metric: f1-score
direction: maximize
data_dir: dataorg-financial-health-prediction-challenge
---
Can you predict the financial well-being of small businesses? (current best performance I got without you is 0.895390236 F1 Score on LB)

Across Southern Africa, small and medium-sized enterprises (SMEs) are vital to employment, innovation, and economic growth, yet many remain financially fragile and excluded from formal financial systems. Limited access to credit, unstable cash flow, and exposure to shocks such as illness or climate events make them vulnerable. Traditional measures like revenue or profit do not capture an SME’s true financial well-being. To support SMEs more effectively, there is a need for a holistic measure that reflects resilience, savings habits, and access to finance.

This challenge introduces a data-driven Financial Health Index (FHI) for SMEs - a composite measure that classifies businesses into Low, Medium, or High financial health across four key dimensions: savings and assets, debt and repayment ability, resilience to shocks, and access to credit and financial services. Derived from survey and business data, the FHI offers a more complete picture of financial stability and inclusion.

Participants will build machine learning models to predict the FHI using socio-economic and business data such as traded commodities, export and import activity, demographics, firm size, and location. This data is sourced from four Southern African countries - Eswatini, Lesotho, Zimbabwe, and Malawi. But the relevance of such an index extends to businesses in developing economies all over the world.

By quantifying SME financial health, the challenge supports data-driven policies and inclusive financing strategies. Financial institutions can better assess credit risk, while development partners and governments can identify vulnerable businesses and target support where it is needed most.

Ultimately, the Financial Health Index redefines how SME wellbeing is measured, beyond profits to resilience and opportunity. By predicting how financially healthy a business is today, participants will help shape the tools and insights that enable small enterprises to thrive tomorrow.

Learning Resources

To help you get started with data for financial inclusion, we have partnered with the [Capacity Accelerator Network (CAN)](https://data.org/initiatives/capacity/) powered by [data.org](http://data.org/) to provide free online accreditation in some relevant courses:. These courses are quick to complete and will provide you with the skills and inspiration you need to succeed in this challenge.

* [Fintech Literacy Program](https://zindi.africa/courses/course/OvMU4N/modules)
* [Introduction to Responsible Data Management](https://zindi.africa/courses/course/NolU2O/modules)
* [Ethical AI in Practice](https://zindi.africa/courses/course/NnWUdO/modules)

![](https://zindi-public-release.s3.eu-west-2.amazonaws.com/uploads/image_attachment/image/3116/16514dd4-8570-4a14-91b9-27aff4424846.png)

[About data.org](https://data.org/)

data.org is a platform accelerating the use of data and AI to solve major global challenges and build the field of data for social impact. It convenes and coordinates across sectors to advance practical solutions, host innovation challenges, and train purpose-driven data practitioners. By 2032, data.org aims to train 1 million purpose-driven data and AI professionals, foster digital public goods, and build connections for impactful data use around the world.

[About FinMark Trust](https://finmark.org.za/)

FinMark Trust is an independent non-profit trust with the purpose of making financial markets work for people in poverty by promoting financial inclusion and regional financial integration. We pursue our core objective of making financial markets work for people living in poverty through two principal programmes. The first happens through the creation and analysis of financial services demand-side data to provide in-depth insights on both served and unserved consumers across the developing world. The second is through systematic financial sector inclusion and deepening programmes to overcome regulatory, supplier and other market-level barriers hampering the effective provision of services.

[About Indai AI Impact Summit 2026](https://impact.indiaai.gov.in/)

The India–AI Impact Summit 2026, announced by Hon’ble Prime Minister Narendra Modi at the France AI Action Summit and scheduled for February 19–20 in New Delhi, will be the first-ever global AI summit hosted in the Global South. Building on the momentum of leading international forums such as the UK AI Safety Summit, the AI Seoul Summit, the France AI Action Summit, and the Global AI Summit on Africa, this high-level convening marks a critical inflection point. It will strengthen existing multilateral initiatives while advancing new priorities, deliverables, and cooperative frameworks—moving from high-level political statements to demonstrable impact and tangible progress in global AI cooperation.

Evaluation

The evaluation metric for this challenge is the [F1 score](https://zindi.africa/learn/zindi-error-metric-series-how-to-use-the-f1-score).

For every row in the dataset, submission files should contain 2 columns: ID and Target:

```
ID            Target
```

```
ID_5EGLKX    Low
```

```
ID_4AI7RE    Low
```

```
ID_V9OB3M    Low
```

Prizes

1st place: $750 USD

2nd place: $500 USD

3rd place: $250 USD

There are 3 000 Zindi points available. You can read more about [Zindi points here](https://zindi.africa/discussions/13959?utm_source=zindi&utm_medium=blog&utm_campaign=challenge_resources&utm_id=CR).

Rules

* Languages and tools: You may only use open-source languages and tools in building models for this challenge.
* Who can compete: Open to all
* Submission Limits: 10 submissions per day, 200 submissions overall.
* Team size: Max team size of 4
* Public-Private Split: Zindi maintains a public leaderboard and a private leaderboard for each challenge. The Public Leaderboard includes approximately 30% of the test dataset. The private leaderboard will be revealed at the close of the challenge and contains the remaining 70% of the test set.
* Data Sharing: CC-BY SA 4.0 license
* Code Review: Top 10 on the private leaderboard will receive an email requesting their code at the close of the challenge. You will have 48 hours to submit your code.
* Code sharing: Multiple accounts, or sharing of code and information across accounts not in teams, is not allowed and will lead to disqualification.

ENTRY INTO THIS CHALLENGE CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL CHALLENGE RULES.

Full Challenge Rules

This challenge is open to all.

Teams and collaboration

You may participate in challenges as an individual or in a team of up to four people. When creating a team, the team must have a total submission count less than or equal to the maximum allowable submissions as of the formation date. A team will be allowed the maximum number of submissions for the challenge, minus the total number of submissions among team members at team formation. Prizes are transferred only to the individual players or to the team leader.

Multiple accounts per user are not permitted, and neither is collaboration or membership across multiple teams. Individuals and their submissions originating from multiple accounts will be immediately disqualified from the platform.

Code must not be shared privately outside of a team. Any code that is shared, must be made available to all challenge participants through the platform. (i.e. on the discussion boards).

The Zindi data scientist who sets up a team is the default Team Leader but they can transfer leadership to another data scientist on the team. The Team Leader can invite other data scientists to their team. Invited data scientists can accept or reject invitations. Until a second data scientist accepts an invitation to join a team, the data scientist who initiated a team remains an individual on the leaderboard. No additional members may be added to teams within the final 5 days of the challenge or last hour of a hackathon.

The team leader can initiate a merge with another team. Only the team leader of the second team can accept the invite. The default team leader is the leader from the team who initiated the invite. Teams can only merge if the total number of members is less than or equal to the maximum team size of the challenge.

A team can be disbanded if it has not yet made a submission. Once a submission is made individual members cannot leave the team.

All members in the team receive points associated with their ranking in the challenge and there is no split or division of the points between team members.

Datasets, packages and general principles

The solution must use publicly-available, open-source packages only.

You may use only the datasets provided for this challenge.

You may use pretrained models as long as they are openly available to everyone.

Automated machine learning tools such as automl are not permitted.

If the error metric requires probabilities to be submitted, do not set thresholds (or round your probabilities) to improve your place on the leaderboard. In order to ensure that the client receives the best solution Zindi will need the raw probabilities. This will allow the clients to set thresholds to their own needs.

You are allowed to access, use and share challenge data for any commercial, non-commercial, research or education purposes, under a CC-BY SA 4.0 license.

You must notify Zindi immediately upon learning of any unauthorised transmission of or unauthorised access to the challenge data, and work with Zindi to rectify any unauthorised transmission or access.

Your solution must not infringe the rights of any third party and you must be legally entitled to assign ownership of all rights of copyright in and to the winning solution code to Zindi.

Submissions and winning

You may make a maximum of 10 submissions per day.

You may make a maximum of 300 submissions for this challenge.

Before the end of the challenge you need to choose 2 submissions to be judged on for the private leaderboard. If you do not make a selection your 2 best public leaderboard submissions will be used to score on the private leaderboard.

During the challenge, your best public score will be displayed regardless of the submissions you have selected. When the challenge closes your best private score out of the 2 selected submissions will be displayed.

Zindi maintains a public leaderboard and a private leaderboard for each challenge. The Public Leaderboard includes approximately 20% of the test dataset. While the challenge is open, the Public Leaderboard will rank the submitted solutions by the accuracy score they achieve. Upon close of the challenge, the Private Leaderboard, which covers the other 80% of the test dataset, will be made public and will constitute the final ranking for the challenge.

Note that to count, your submission must first pass processing. If your submission fails during the processing step, it will not be counted and not receive a score; nor will it count against your daily submission limit. If you encounter problems with your submission file, your best course of action is to ask for advice on the challenge page.

If you are in the top 10 at the time the leaderboard closes, we will email you to request your code. On receipt of email, you will have 48 hours to respond and submit your code following the Reproducibility of submitted code guidelines detailed below. Failure to respond will result in disqualification.

If your solution places 1st, 2nd, or 3rd on the final leaderboard, you will be required to submit your winning solution code to us for verification, and you thereby agree to assign all worldwide rights of copyright in and to such winning solution to Zindi.

If two solutions earn identical scores on the leaderboard, the tiebreaker will be the date and time in which the submission was made (the earlier solution will win).

The winners will be paid via bank transfer, PayPal if payment is less than or equivalent to $100, or other international money transfer platform. International transfer fees will be deducted from the total prize amount, unless the prize money is under $500, in which case the international transfer fees will be covered by Zindi. In all cases, the winners are responsible for any other fees applied by their own bank or other institution for receiving the prize money. All taxes imposed on prizes are the sole responsibility of the winners. The top winners or team leaders will be required to present Zindi with proof of identification, proof of residence and a letter from your bank confirming your banking details. Winners will be paid in USD or the currency of the challenge. If your account cannot receive US Dollars or the currency of the challenge then your bank will need to provide proof of this and Zindi will try to accommodate this.

Please note that due to the ongoing Russia-Ukraine conflict, we are not currently able to make prize payments to winners located in Russia. We apologise for any inconvenience that may cause, and will handle any issues that arise on a case-by-case basis.

Payment will be made after code review and sealing the leaderboard.

You acknowledge and agree that Zindi may, without any obligation to do so, remove or disqualify an individual, team, or account if Zindi believes that such individual, team, or account is in violation of these rules. Entry into this challenge constitutes your acceptance of these official challenge rules.

Zindi is committed to providing solutions of value to our clients and partners. To this end, we reserve the right to disqualify your submission on the grounds of usability or value. This includes but is not limited to the use of data leaks or any other practices that we deem to compromise the inherent value of your solution.

Zindi also reserves the right to disqualify you and/or your submissions from any challenge if we believe that you violated the rules or violated the spirit of the challenge or the platform in any other way. The disqualifications are irrespective of your position on the leaderboard and completely at the discretion of Zindi.

Please refer to the FAQs and Terms of Use for additional rules that may apply to this challenge. We reserve the right to update these rules at any time.

Reproducibility of submitted code

If your submitted code does not reproduce your score on the leaderboard, we reserve the right to adjust your rank to the score generated by the code you submitted.

If your code does not run you will be dropped from the top 10. Please make sure your code runs before submitting your solution.

Always set the seed. Rerunning your model should always place you at the same position on the leaderboard. When running your solution, if randomness shifts you down the leaderboard we reserve the right to adjust your rank to the closest score that your submission reproduces.

Custom packages in your submission notebook will not be accepted.

You may only use tools available to everyone i.e. no paid services or free trials that require a credit card.

Read [this article](https://zindi.africa/learn/documentation-guideline) on how to prepare your documentation and[ this article](https://zindi.africa/learn/how-to-ensure-success-when-submitting-your-code-for-review) on how to ensure a successful code review.

Consequences of breaking any rules of the challenge or submission guidelines:

* First offence: No prizes for6 months and 2000 points will be removed from your profile (probation period). If you are caught cheating, all individuals involved in cheating will be disqualified from the challenge(s) you were caught in and you will be disqualified from winning any challenges for the next six months and 2000 points will be removed from your profile. If you have less than 2000 points to your profile your points will be set to 0.
* Second offence: Banned from the platform. If you are caught for a second time your Zindi account will be disabled and you will be disqualified from winning any challenges or Zindi points using any other account.

Teams with individuals who are caught cheating will not be eligible to win prizes or points in the challenge in which the cheating occurred, regardless of the individuals’ knowledge of or participation in the offence.

Teams with individuals who have previously committed an offence will not be eligible for any prizes for any challenges during the 6-month probation period.

Monitoring of submissions

We will review the top 10 solutions of every challenge when the challenge ends.

We reserve the right to request code from any user at any time during a challenge. You will have 24 hours to submit your code following the rules for code review (see above). Zindi reserves the right not to explain our reasons for requesting code. If you do not submit your code within 24 hours you will be disqualified from winning any challenges or Zindi points for the next six months. If you fall under suspicion again and your code is requested and you fail to submit your code within 24 hours, your Zindi account will be disabled and you will be disqualified from winning any challenges or Zindi points with any other account.

About

This challenge provides a rich dataset capturing the financial behaviour, resilience, and operational realities of small and medium-sized enterprises (SMEs) across Eswatini, Lesotho, Malawi, and Zimbabwe. The data is sourced from SME surveys and includes detailed information about business owners, their financial habits, exposure to risks, access to credit, and overall business performance.

Your task is to use these features to predict the Financial Health Index (FHI) - a category of low, medium or high that reflects how financially resilient and well-positioned each business is. The index incorporates aspects such as savings habits, debt, shock resilience, and access to formal financial services.

Files

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/dataorg-financial-health-prediction-challenge20251204-19827-m2tn1n.zip?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)Download all files in archive

649.3 KB

Description

Files

Test resembles Train.csv but without the target column. This is the dataset on which you will apply your model to.

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/Test.csv?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)Test.csv

506.4 KB

This is the dataset that you will use to train your model. It contains the target.

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/Train.csv?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)Train.csv

2 MB

This shows the submission format for this challenge.

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/SampleSubmission.csv?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)SampleSubmission.csv

35.2 KB

Full list of variables and their explanations.

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/VariableDefinitions.csv?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)VariableDefinitions.csv

3.3 KB

This is a starter notebook to help you make your first submission.

[](https://api.zindi.africa/v1/competitions/dataorg-financial-health-prediction-challenge/files/Starter%20Notebook.ipynb?auth_token=zus.v1.D4SLxOE.NpEaXGHUiN7Q8cTm9qyzoqYwRbDsuJ)Starter Notebook.ipynb

486.6 KB
