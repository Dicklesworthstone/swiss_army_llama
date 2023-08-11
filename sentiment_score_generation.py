from llama_2_embeddings_fastapi_server import load_token_level_embedding_model
from llama_2_embeddings_fastapi_server import configured_logger as logger
import asyncio
import psutil
import glob
import os
import sys
import re
import json
import traceback
import numpy as np
import pandas as pd
from filelock import FileLock
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from llama_cpp import Llama
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


financial_investor_focus_presets = """
    Optimistic: Reflecting a positive outlook or confidence in future success.
    Pessimistic: Expressing a lack of hope or a negative view of future prospects.
    Confident: Demonstrating self-assurance and belief in strategies or decisions.
    Uncertain: Highlighting doubt or lack of clarity regarding future events.
    Positive: Conveying a favorable or encouraging sentiment.
    Negative: Indicating an unfavorable or discouraging sentiment.
    Cautious: Showing carefulness, often related to risks or potential challenges.
    Aggressive: Reflecting a forceful or assertive approach, often in strategy.
    Satisfied: Expressing contentment or approval with current conditions.
    Disappointed: Showing dissatisfaction or displeasure with outcomes.
    Enthusiastic: Conveying strong excitement or eagerness about something.
    Worried: Reflecting concern or anxiety about potential problems.
    Transparent: Indicating openness and clarity in communication.
    Ambiguous: Conveying uncertainty or vagueness, often leading to confusion.
    Stable: Reflecting steadiness, consistency, or lack of volatility.
    Volatile: Indicating instability, unpredictability, or rapid changes.
    Committed: Demonstrating dedication or strong allegiance to a cause or plan.
    Indifferent: Showing lack of interest, concern, or importance.
    Receptive: Indicating willingness to consider new ideas or feedback.
    Defensive: Reflecting a protective stance, often in response to criticism or challenges.
"""  # noqa: E501

public_company_earnings_call_text = """
    Optimistic: Expressing confidence in future growth, profitability, or success.
    Pessimistic: Showing concern or lack of confidence in future prospects.
    Confident: Demonstrating strong belief in strategies, products, or market position.
    Cautious: Reflecting careful consideration, especially regarding risks or challenges.
    Transparent: Providing clear and open information, often related to finances or strategies.
    Vague: Lacking clarity or specific details, possibly to avoid commitment.
    Upbeat: Conveying a positive and enthusiastic tone about performance or opportunities.
    Disappointed: Expressing dissatisfaction with results, outcomes, or specific areas.
    Reassuring: Offering assurances or comfort regarding concerns or uncertainties.
    Evasive: Avoiding direct answers or clear statements on sensitive topics.
    Committed: Demonstrating dedication to initiatives, strategies, or goals.
    Analytical: Providing a detailed and thoughtful analysis of performance and trends.
    Ambitious: Reflecting strong desire for growth, innovation, or market leadership.
    Concerned: Showing worry or apprehension, often related to external factors or challenges.
    Focused: Emphasizing specific priorities, goals, or areas of attention.
    Uncertain: Highlighting ambiguity or lack of clarity on specific matters or outlook.
    Responsive: Addressing questions or concerns promptly and thoroughly.
    Defensive: Protecting or justifying decisions, results, or strategies.
    Strategic: Emphasizing long-term planning, direction, and competitive positioning.
    Realistic: Providing a grounded and practical perspective on performance and expectations.
"""

customer_feedback_focus_presets = """
    Satisfied: Expressing contentment or approval with the product or service.
    Dissatisfied: Showing dissatisfaction or displeasure with the product or service.
    Impressed: Reflecting a strong positive impression or admiration.
    Frustrated: Indicating irritation or annoyance, often due to unmet expectations.
    Appreciative: Conveying gratitude or appreciation for a particular aspect.
    Confused: Reflecting uncertainty or confusion about a feature or experience.
    Delighted: Showing great pleasure, satisfaction, or happiness.
    Disappointed: Expressing discontentment or letdown with the overall experience.
    Concerned: Reflecting worry or concern about a specific issue or experience.
    Excited: Conveying anticipation or eagerness about a new feature or update.
    Annoyed: Indicating minor irritation or displeasure with an aspect of the service.
    Surprised: Reflecting an unexpected experience, either positive or negative.
    Comfortable: Indicating ease or comfort with using the product or service.
    Inconvenienced: Reflecting a sense of hassle or difficulty in interaction.
    Trusting: Demonstrating confidence or trust in the brand or product.
    Skeptical: Showing doubt or skepticism towards claims or promises.
    Loyal: Reflecting a strong allegiance or loyalty to the brand or product.
    Indifferent: Conveying a lack of strong feelings one way or another.
    Curious: Indicating an interest or curiosity about future developments.
    Unsupported: Feeling a lack of support or assistance when needed.
"""

customer_service_interaction_focus_presets = """
    Empathetic: Showing understanding and compassion for the customer's situation.
    Unsympathetic: Lacking concern or understanding for the customer's needs.
    Helpful: Providing valuable assistance, guidance, or solutions.
    Unhelpful: Failing to provide meaningful support or resolution.
    Patient: Taking time to listen and understand the customer's concerns.
    Impatient: Showing frustration or haste, possibly dismissing the customer's needs.
    Clear: Communicating information in an easy-to-understand and straightforward manner.
    Confusing: Providing information that may lead to misunderstandings or confusion.
    Responsive: Reacting promptly and appropriately to questions or concerns.
    Slow: Delaying responses or taking excessive time to resolve issues.
    Friendly: Expressing warmth, politeness, and a welcoming tone.
    Cold: Reflecting a lack of warmth, possibly perceived as indifferent or distant.
    Professional: Maintaining a respectful and business-like demeanor.
    Informal: Using casual language or tone, possibly perceived as unprofessional.
    Reassuring: Offering comfort or confidence that issues will be resolved.
    Dismissive: Indicating a lack of interest or disregarding the customer's concerns.
    Appreciative: Expressing gratitude for the customer's patience or feedback.
    Apologetic: Offering apologies for mistakes or inconveniences.
    Inquisitive: Asking questions to clarify or understand the customer's needs.
    Uninterested: Showing lack of engagement or interest in assisting the customer.
"""

marketing_campaign_focus_presets = """
    Persuasive: Effectively convincing or influencing the audience.
    Engaging: Capturing interest and encouraging interaction or participation.
    Innovative: Introducing new ideas or creative approaches.
    Clichéd: Relying on overused or predictable concepts and phrases.
    Inspirational: Motivating or uplifting, often evoking emotional responses.
    Confusing: Lacking clarity or causing misunderstandings in messaging.
    Aggressive: Using forceful or assertive tactics to promote a product or idea.
    Subtle: Communicating messages in a less obvious or understated manner.
    Authentic: Conveying a genuine or sincere brand image or message.
    Misleading: Providing information that may deceive or create false impressions.
    Inclusive: Emphasizing diversity and broad appeal to various audiences.
    Exclusive: Targeting a specific or niche audience, possibly excluding others.
    Visually_Stunning: Utilizing impressive visual design or imagery.
    Monotonous: Lacking variety or excitement in content or presentation.
    Responsive: Adapting to audience feedback or market trends.
    Rigid: Sticking strictly to a specific concept or style, lacking flexibility.
    Emotional: Evoking strong emotional reactions or connections.
    Factual: Relying on facts, data, or evidence in messaging.
    Trendy: Aligning with current trends or popular culture.
    Timeless: Conveying messages that are enduring and not tied to temporary trends.
"""

product_review_focus_presets = """
    Satisfied: Expressing contentment or approval with the product's quality or performance.
    Dissatisfied: Showing dissatisfaction or displeasure with the product's attributes.
    Impressed: Reflecting a positive impression or admiration for the product.
    Disappointed: Expressing discontentment or letdown with the overall experience.
    Excited: Conveying enthusiasm or eagerness about the product's features or design.
    Frustrated: Indicating irritation or annoyance with the product's functionality or support.
    Valuable: Perceiving the product as worth the price or offering good value.
    Overpriced: Reflecting the belief that the product is too expensive for what it offers.
    Innovative: Recognizing the product as new, creative, or cutting-edge.
    Outdated: Viewing the product as old-fashioned or behind current trends.
    User-friendly: Finding the product easy to use or intuitive.
    Complicated: Experiencing difficulty or confusion in using the product.
    Reliable: Trusting the product to perform consistently and without issues.
    Unreliable: Encountering inconsistencies or problems in the product's performance.
    Attractive: Appreciating the product's appearance or aesthetic design.
    Unappealing: Finding the product's appearance or design unattractive or displeasing.
    Essential: Viewing the product as a must-have or vital to specific needs.
    Redundant: Perceiving the product as unnecessary or superfluous.
    High-Quality: Recognizing the product's superior craftsmanship or materials.
    Low-Quality: Identifying flaws or shortcomings in the product's build or design.
"""

email_correspondence_focus_presets = """
    Formal: Adhering to conventional or traditional standards in language and tone.
    Informal: Using a casual or conversational tone.
    Courteous: Expressing politeness, respect, or consideration.
    Abrupt: Conveying a sudden or brusque tone, possibly perceived as rude.
    Clear: Providing information that is easy to understand and unambiguous.
    Confusing: Containing elements that may lead to misunderstanding or confusion.
    Persuasive: Intending to convince or influence the recipient.
    Indifferent: Lacking interest or concern in the subject matter.
    Appreciative: Expressing gratitude or thanks.
    Apologetic: Conveying regret or offering an apology for something.
    Enthusiastic: Showing excitement or strong interest in the topic.
    Disinterested: Reflecting a lack of personal interest or bias.
    Urgent: Conveying a sense of immediacy or time-sensitivity.
    Nonchalant: Reflecting a lack of concern or appearing casually unconcerned.
    Encouraging: Offering support or motivation.
    Critical: Expressing criticism or pointing out faults.
    Responsive: Providing a timely and relevant reply.
    Unresponsive: Lacking a reply or failing to address the subject.
    Inquisitive: Asking questions or seeking more information.
    Directive: Providing clear instructions or directions.
"""

github_issues_focus_presets = """
    Constructive: Providing useful feedback or contributing positively to the discussion.
    Critical: Pointing out problems or issues, possibly in a negative tone.
    Appreciative: Expressing gratitude or appreciation for work done.
    Frustrated: Indicating irritation or annoyance with a bug or lack of progress.
    Urgent: Conveying a sense of immediacy or importance in resolving an issue.
    Confused: Reflecting uncertainty or confusion about code, features, or requirements.
    Collaborative: Showing a willingness to work together or assist others.
    Dismissive: Indicating a lack of interest or disregard for the issue or comment.
    Informative: Providing detailed information, context, or guidance.
    Vague: Lacking clarity or specific details, possibly leading to misunderstandings.
    Supportive: Offering assistance, encouragement, or backing for an idea or request.
    Argumentative: Engaging in disagreement or debate, possibly in a confrontational manner.
    Responsive: Reacting promptly and appropriately to questions or concerns.
    Unresponsive: Failing to provide timely or relevant responses.
    Encouraging: Motivating or inspiring further work or discussion.
    Inquisitive: Seeking more information, asking clarifying questions.
    Detailed: Providing a thorough and comprehensive explanation or description.
    Concise: Communicating information in a clear and brief manner.
    Receptive: Open to feedback, suggestions, or new ideas.
    Defensive: Reacting protectively, possibly in response to criticism or feedback.
"""

social_media_sentiment_focus_presets = """
    Positive: Expressing favorable opinions or support.
    Negative: Conveying criticism or opposition.
    Neutral: Lacking strong emotion or bias; impartial.
    Engaging: Capturing interest; encouraging interaction or response.
    Polarizing: Provoking strong disagreement or contrasting opinions.
    Inspirational: Providing motivation or encouragement.
    Sarcastic: Conveying irony or mocking insincerity.
    Humorous: Intending to entertain or amuse.
    Controversial: Addressing subjects likely to arouse strong opinions or debate.
    Supportive: Offering sympathy, encouragement, or assistance.
    Hostile: Expressing aggression, antagonism, or conflict.
    Curious: Seeking information or expressing genuine interest.
    Informative: Providing useful information or insights.
    Emotional: Reflecting strong emotions or feelings.
    Detached: Lacking emotional involvement or personal bias.
    Celebratory: Expressing joy, congratulations, or festivity.
    Mourning: Expressing grief, sorrow, or remembrance.
    Respectful: Showing consideration and regard for others.
    Disrespectful: Lacking courtesy; rude or offensive.
    Trendy: Reflecting current trends, fads, or popular culture.
"""

employee_feedback_focus_presets = """
    Positive: Conveying encouragement, praise, or satisfaction with performance.
    Negative: Indicating dissatisfaction, criticism, or areas for improvement.
    Constructive: Providing actionable insights and suggestions for growth.
    Supportive: Showing empathy, understanding, and encouragement.
    Disengaged: Reflecting a lack of interest or connection with the employee's situation.
    Appreciative: Expressing gratitude or recognition for efforts and achievements.
    Concerned: Reflecting worry or unease about performance or behavior.
    Fair: Providing balanced and unbiased feedback.
    Biased: Showing favoritism or prejudice, affecting the objectivity of feedback.
    Motivational: Inspiring or encouraging further growth and development.
    Demotivational: Discouraging or undermining confidence and enthusiasm.
    Specific: Providing clear and detailed insights or directions.
    Vague: Lacking clarity or specificity, making it hard to act upon.
    Respectful: Treating the employee with dignity and consideration.
    Disrespectful: Conveying a lack of respect or courtesy.
    Empathetic: Showing understanding and compassion for the employee's feelings or situation.
    Impersonal: Lacking warmth or personal connection, possibly feeling mechanical.
    Engaging: Encouraging dialogue, questions, and active participation.
    Confident: Expressing strong belief in decisions, assessments, or directions.
    Uncertain: Lacking conviction or clarity in the feedback provided.
"""

crisis_communication_focus_presets = """
    Urgent: Conveying a sense of immediate need or priority.
    Reassuring: Providing comfort or confidence in the face of uncertainty.
    Clear: Communicating information in an unambiguous and straightforward manner.
    Confusing: Lacking clarity or leading to misunderstandings.
    Calm: Reflecting a composed or steady demeanor.
    Panicked: Showing signs of fear or anxiety, possibly leading to hasty decisions.
    Empathetic: Demonstrating understanding and compassion for those affected.
    Detached: Lacking emotion or personal connection, possibly perceived as cold.
    Informative: Providing essential details, updates, or guidance.
    Vague: Being unclear or lacking specific information, causing uncertainty.
    Responsive: Reacting promptly and appropriately to evolving situations.
    Slow: Delaying responses or updates, possibly leading to frustration.
    Honest: Conveying truthful and transparent information.
    Evasive: Avoiding direct answers or withholding information.
    Authoritative: Speaking with command and expertise, instilling trust.
    Hesitant: Showing uncertainty or reluctance in communication.
    Encouraging: Inspiring hope or confidence during a challenging time.
    Alarming: Causing unnecessary fear or concern, possibly due to exaggerated language.
    Collaborative: Working together with others, coordinating efforts.
    Defensive: Reacting protectively or defensively, often in response to criticism.
"""

political_speech_focus_presets = """
    Inspirational: Providing motivation and encouragement, often appealing to shared values or goals.
    Divisive: Creating or emphasizing division or disagreement among different groups.
    Diplomatic: Expressing ideas in a sensitive, tactful, and balanced manner.
    Aggressive: Using forceful or confrontational language to make a point.
    Patriotic: Expressing love, support, or devotion to one's country.
    Critical: Pointing out problems or criticizing opponents or policies.
    Visionary: Outlining a future goal or plan in an imaginative or idealistic way.
    Defensive: Responding to criticism or defending a position or policy.
    Conciliatory: Seeking to reconcile differences and bring about harmony or agreement.
    Evocative: Eliciting strong emotions or memories, often through vivid language.
    Rational: Based on logic and reason, often using facts and evidence to support arguments.
    Emotional: Appealing to the emotions, possibly using anecdotes or passionate language.
    Authoritative: Speaking with confidence and authority, often citing expertise or credentials.
    Populist: Appealing to the concerns and interests of ordinary people.
    Evasive: Avoiding direct answers or clear statements on specific issues.
    Respectful: Showing regard and consideration for others, even when disagreeing.
    Condescending: Talking down to the audience or treating them as inferior or less knowledgeable.
    Unifying: Seeking to bring people together, emphasizing common goals or values.
    Provocative: Intended to provoke thought or reaction, possibly through controversial statements.
    Compassionate: Expressing sympathy or concern for the needs and suffering of others.
"""

healthcare_patient_feedback_focus_presets = """
    Satisfied: Expressing contentment or approval with the care received.
    Dissatisfied: Showing dissatisfaction or displeasure with healthcare services.
    Grateful: Conveying gratitude or appreciation for medical assistance.
    Anxious: Reflecting anxiety or concern about a diagnosis, treatment, or recovery.
    Confident: Demonstrating trust or confidence in healthcare providers.
    Uncertain: Expressing doubt or confusion about medical information or decisions.
    Compassionate: Reflecting empathy, warmth, or understanding from healthcare providers.
    Neglected: Feeling overlooked or lacking attention from medical staff.
    Informed: Feeling well-informed and educated about medical conditions and treatments.
    Misinformed: Receiving incorrect or misleading information.
    Comfortable: Indicating physical or emotional comfort during medical care.
    Uncomfortable: Reflecting physical discomfort or unease with medical procedures.
    Hopeful: Conveying optimism or hope regarding recovery or treatment outcomes.
    Desperate: Expressing a strong need or desire for immediate help or relief.
    Supported: Feeling supported or cared for by healthcare providers.
    Unsupported: Lacking support or empathy from medical staff.
    Empowered: Feeling control or influence over medical decisions and care.
    Vulnerable: Feeling exposed or at risk, possibly due to health conditions or treatments.
    Appreciative: Expressing thanks or positive recognition for care provided.
    Disheartened: Feeling discouraged or dispirited about medical outcomes or experiences.
"""

educational_content_focus_presets = """
    Informative: Providing valuable information or insights.
    Confusing: Lacking clarity or causing misunderstandings.
    Engaging: Capturing interest and encouraging active participation.
    Dull: Lacking excitement or interest; monotonous.
    Challenging: Presenting difficulties or stimulating intellectual effort.
    Simplistic: Overly simple or lacking depth.
    Comprehensive: Covering a subject thoroughly or in detail.
    Superficial: Lacking depth or detailed coverage of a topic.
    Relevant: Directly related to the subject or current trends.
    Outdated: Containing information that is no longer current or applicable.
    Practical: Providing hands-on experience or applicable knowledge.
    Theoretical: Focusing on theories or abstract concepts without practical application.
    Inspiring: Encouraging creativity, motivation, or a positive attitude toward learning.
    Discouraging: Dampening interest or enthusiasm in the subject.
    Balanced: Presenting multiple perspectives or a fair representation of a subject.
    Biased: Showing prejudice or favoritism toward a particular view or group.
    Interactive: Encouraging active participation or engagement with the content.
    Passive: Requiring little or no interaction or engagement.
    Structured: Organized and logically arranged for ease of understanding.
    Disorganized: Lacking clear structure or logical sequence.
"""

job_interview_focus_presets = """
    Professional: Adhering to formal language, attire, and conduct.
    Casual: Adopting a relaxed and informal approach or tone.
    Inquisitive: Asking detailed and probing questions to understand the candidate's experience and thinking.
    Superficial: Sticking to surface-level questions without digging into depth or details.
    Technical: Focusing on specific skills, tools, or methodologies relevant to the job.
    Behavioral: Concentrating on attitudes, ethics, and interpersonal skills.
    Motivational: Exploring the candidate's passions, goals, and what drives them.
    Structured: Following a set pattern or sequence of questions and activities.
    Unstructured: Lacking a clear pattern, possibly leading to a more organic and free-flowing conversation.
    Friendly: Creating a warm and welcoming atmosphere to make the candidate comfortable.
    Intimidating: Taking a more aggressive or challenging approach that might test the candidate's ability to handle pressure.
    Balanced: Providing a fair mix of technical, behavioral, and motivational questions.
    Biased: Showing favoritism or prejudice towards certain backgrounds, experiences, or views.
    Transparent: Clearly communicating the interview process, expectations, and any feedback.
    Confidential: Emphasizing the privacy of the candidate's responses and information.
    Encouraging: Building confidence and enthusiasm during the interview process.
    Discouraging: Creating a negative or disheartening atmosphere, possibly through overly critical or harsh questioning.
    Adaptive: Tailoring questions and approach based on the candidate's responses and unique background.
    Rigid: Sticking strictly to predetermined questions without flexibility or adaptation to the candidate's specific situation.
    Holistic: Considering all aspects of the candidate, including skills, personality, values, and fit with the organization's culture.
    Narrow: Focusing on one or a few specific areas, potentially overlooking a broader view of the candidate's suitability.
"""

sales_call_focus_presets = """
    Persuasive: Utilizing convincing language and arguments to promote the product or service.
    Informative: Focusing on providing detailed information about the product or service.
    Aggressive: Employing a forceful and assertive approach to secure a sale.
    Consultative: Engaging in a dialogue to understand needs and offering tailored solutions.
    Scripted: Following a predetermined script or set of talking points.
    Spontaneous: Adapting and reacting to the prospect's responses without strict adherence to a script.
    Friendly: Creating a warm and approachable atmosphere to build rapport.
    Formal: Maintaining a professional tone and language aligned with corporate standards.
    Solution-Oriented: Concentrating on how the product or service solves specific problems or fulfills needs.
    Feature-Focused: Highlighting the unique features and specifications of the product or service.
    Customer-Centric: Tailoring the conversation around the customer's specific needs and interests.
    Product-Centric: Centering the conversation mainly on the product or service being offered, possibly overlooking customer needs.
    Transparent: Clearly communicating pricing, terms, and any relevant details without hidden agendas.
    Evasive: Avoiding direct answers or details, possibly creating confusion or mistrust.
    Collaborative: Working with the prospect to find the best solution, encouraging their input and feedback.
    Competitive: Emphasizing why the product or service is superior to competitors.
    Empathetic: Showing understanding and compassion for the prospect's situation or needs.
    Time-Sensitive: Creating urgency through limited-time offers or emphasizing immediate needs.
    Patient: Taking the time to listen, explain, and guide without rushing the prospect.
    Closing-Focused: Prioritizing closing the deal, possibly at the expense of building a relationship or understanding needs.
    Relationship-Building: Emphasizing long-term relationships over immediate sales, investing time in understanding and connecting with the prospect.
"""

real_estate_listing_focus_presets = """
    Luxurious: Highlighting high-end finishes, amenities, and exclusive features.
    Affordable: Emphasizing the value or competitive pricing of the property.
    Family-Friendly: Targeting families by showcasing features like spacious rooms, yards, and proximity to schools.
    Modern: Focusing on contemporary design, technology, and aesthetics.
    Vintage: Celebrating historical or traditional architectural features and charm.
    Eco-Friendly: Promoting energy efficiency, sustainability, or environmentally friendly aspects.
    Investment-Oriented: Tailoring the listing to appeal to investors, emphasizing rental income or appreciation potential.
    Location-Centric: Concentrating on the location's benefits, such as convenience, views, or prestige.
    Community-Focused: Highlighting community amenities, neighbors, or local culture and vibe.
    Practical: Providing clear and concise information about the property without embellishments.
    Visual: Relying heavily on high-quality images, videos, or virtual tours to showcase the property.
    Detailed: Offering an exhaustive description, including every feature, measurement, and specification.
    Minimalist: Using a brief and to-the-point description, relying on key selling points.
    Pet-Friendly: Emphasizing features that cater to pet owners, such as a fenced yard or nearby parks.
    Accessible: Highlighting accessibility features for individuals with disabilities or mobility challenges.
    Renovation Potential: Pointing out opportunities for improvements or customization.
    Move-In Ready: Emphasizing that the property is in turnkey condition without need for immediate repairs or updates.
    Resort-Style: Showcasing amenities and features that create a vacation-like living experience.
    Urban: Focusing on the benefits of city living, such as proximity to public transport, dining, and entertainment.
    Rural: Emphasizing the tranquility, space, and natural surroundings typical of countryside properties.
    Transparent: Clearly stating all costs, potential issues, or other critical details that buyers should be aware of.
"""

mental_health_counseling_focus_presets = """
    Empathetic: Demonstrating understanding and compassion for the client's feelings.
    Directive: Providing clear guidance or instructions to help the client make changes.
    Non-Directive: Allowing the client to lead the direction of the therapy.
    Supportive: Offering encouragement and reassurance as the client explores feelings.
    Challenging: Encouraging the client to confront difficult emotions or behaviors.
    Holistic: Considering all aspects of the client's life, including physical, mental, and social factors.
    Solution-Focused: Concentrating on finding immediate and practical solutions to problems.
    Insight-Oriented: Aiming to increase the client's understanding of themselves and their behaviors.
    Structured: Following a specific therapeutic model or plan.
    Flexible: Adapting the approach to meet the unique needs and preferences of the client.
    Collaborative: Working together with the client to develop goals and strategies.
    Individualized: Tailoring the approach to the specific characteristics and needs of the client.
    Crisis-Intervention: Providing immediate support and guidance for acute mental health crises.
    Long-Term: Focusing on deep and lasting change through extended therapy.
    Short-Term: Offering brief therapy aimed at addressing specific issues or goals.
    Integrative: Combining various therapeutic methods or theories.
    Eclectic: Selecting the best therapy techniques from various approaches, tailored to the client.
    Culturally-Competent: Recognizing and respecting the client's cultural background and values.
    Strength-Based: Focusing on the client's strengths and resources rather than weaknesses.
    Trauma-Informed: Considering the impact of trauma on the client's mental health and well-being.
    Mindfulness-Based: Encouraging present-moment awareness and acceptance.
    Family-Focused: Involving family members in therapy to address relational dynamics.
    Confidential: Ensuring privacy and secrecy of the client's information, in line with ethical guidelines.
    Evidence-Based: Utilizing methods that have been scientifically proven to be effective.
"""

human_resources_recruitment_focus_presets = """
    Inclusive: Promoting diversity and equal opportunity for all candidates.
    Exclusive: Targeting a specific skill set or demographic for specialized roles.
    Transparent: Providing clear and open information about the recruitment process.
    Ambiguous: Lacking clarity in job roles, expectations, or recruitment stages.
    Competitive: Emphasizing a challenging and high-standards selection process.
    Welcoming: Creating a friendly and approachable recruitment environment.
    Structured: Having a well-defined and organized recruitment process.
    Flexible: Allowing for adjustments or accommodations in the recruitment process.
    Traditional: Following conventional recruitment methods and interviews.
    Innovative: Utilizing new and creative ways to attract and assess candidates.
    Candidate-Centric: Focusing on candidate experience and needs.
    Employer-Centric: Emphasizing the company's needs and expectations.
    Ethical: Adhering to fair and moral practices in recruitment.
    Expedient: Prioritizing a quick and efficient recruitment process.
    Thorough: Ensuring detailed evaluation and consideration of candidates.
    Collaborative: Encouraging teamwork and internal alignment in hiring decisions.
    Autonomous: Allowing individual recruiters or managers to make hiring decisions.
    Local: Targeting recruitment efforts in specific geographical areas.
    Global: Expanding recruitment outreach to international talent pools.
    Performance-Based: Assessing candidates primarily on skills and achievements.
    Potential-Based: Evaluating candidates on future potential and growth capacity.
    Legal-Compliant: Ensuring adherence to legal regulations and labor laws.
    Risky: Taking chances on unconventional or unproven recruitment methods.
    Conservative: Adhering to tried-and-tested recruitment principles.
    Growth-Oriented: Focusing on candidates who can contribute to company growth.
    Cultural-Fit: Emphasizing alignment with company culture and values.
"""

political_campaign_focus_presets = """
    Grassroots: Focusing on community-level engagement and local issues.
    Populist: Appealing to the concerns and interests of ordinary people.
    Elitist: Targeting upper-class or highly educated voters with specialized messages.
    Inclusive: Striving to include all demographics and social groups.
    Polarizing: Sharpening divisions and emphasizing differences between groups or parties.
    Unifying: Seeking to bring together disparate groups or interests.
    Progressive: Emphasizing social reform and forward-thinking policies.
    Conservative: Focusing on traditional values and resistance to rapid change.
    Attack-Oriented: Centering on criticism or negative portrayals of opponents.
    Positive: Highlighting own achievements, plans, and constructive ideas.
    Issue-Focused: Concentrating on specific political or social issues.
    Personality-Driven: Relying on the charisma or character of a candidate.
    Pragmatic: Emphasizing practical solutions and realistic goals.
    Idealistic: Promoting visionary goals or utopian ideals.
    Transparent: Providing clear and open disclosure of plans and intentions.
    Ambiguous: Using vague or unclear language to avoid commitment.
    Inspirational: Encouraging hope, motivation, or a positive outlook.
    Fear-Mongering: Using fear or alarmist tactics to gain support.
    Ethical: Emphasizing integrity, honesty, and moral principles.
    Opportunistic: Adapting messages to exploit current events or trends.
    Strategic: Focusing on tactical planning and targeted voter outreach.
    Tactical: Utilizing specific techniques or maneuvers to gain an advantage.
    Cooperative: Encouraging collaboration with other parties or groups.
    Adversarial: Reflecting opposition or conflict with other political entities.
    Localized: Concentrating on regional or local interests and issues.
    Globalized: Addressing international relations and global concerns.
"""

government_policy_response_focus_presets = """
    Proactive: Taking early action to address potential problems or opportunities.
    Reactive: Responding to events or situations as they occur.
    Comprehensive: Addressing all aspects of an issue in a detailed and thorough manner.
    Incremental: Making gradual changes or adjustments to policy.
    Radical: Implementing substantial or fundamental changes to existing structures or policies.
    Conservative: Preserving existing structures and resisting drastic changes.
    Collaborative: Working with other governments, organizations, or stakeholders.
    Unilateral: Acting independently without consultation or cooperation with others.
    Transparent: Providing clear, open disclosure of policy decisions and rationales.
    Opaque: Keeping decisions and rationales hidden or unclear to the public.
    Inclusive: Ensuring that policies consider the needs and interests of all demographic groups.
    Exclusive: Targeting specific groups or interests, potentially at the expense of others.
    Humanitarian: Focusing on the welfare and rights of individuals.
    Economic-Centric: Prioritizing economic growth and fiscal responsibility.
    Environmental: Emphasizing sustainability and protection of natural resources.
    Security-Focused: Concentrating on national security and defense concerns.
    Health-Oriented: Giving priority to public health and wellness.
    Educational: Focusing on educational access, quality, and reform.
    Innovation-Driven: Encouraging technological advancement and creativity.
    Regulatory: Implementing rules and restrictions to govern behavior.
    Deregulatory: Reducing or eliminating governmental rules and oversight.
    Participatory: Encouraging public involvement in policy development or implementation.
    Authoritative: Imposing policies without significant public input or consultation.
    Adaptive: Flexibly adjusting policies in response to changing conditions or feedback.
    Rigid: Sticking to a predetermined policy course regardless of changing circumstances.
    Fair: Ensuring equitable treatment and opportunities for all individuals.
    Biased: Favoring certain groups, interests, or perspectives in policy decisions.
"""

retail_customer_feedback_focus_presets = """
    Product_Quality: Commenting on the quality, durability, or functionality of products.
    Customer_Service: Reflecting on the level of service, friendliness, or responsiveness.
    Price_Value: Assessing the value for money, affordability, or pricing structure.
    Shopping_Experience: Describing the overall experience, ambiance, or layout of the store.
    Convenience: Focusing on the ease of shopping, location, or online accessibility.
    Delivery: Providing feedback on shipping, packaging, or delivery times.
    Product_Selection: Evaluating the variety, availability, or range of products offered.
    Product_Information: Commenting on the clarity, accuracy, or completeness of product descriptions.
    Responsiveness: Rating the speed and effectiveness of responses to inquiries or issues.
    Personalization: Assessing the degree of personalized attention or tailored recommendations.
    Loyalty_Programs: Reflecting on rewards, loyalty schemes, or special offers.
    Ethical_Concerns: Highlighting sustainability, fair trade, or ethical practices.
    Accessibility: Focusing on the accessibility for customers with disabilities or special needs.
    Safety: Commenting on the safety measures, cleanliness, or health protocols.
    Brand_Image: Reflecting on the perception, reputation, or alignment with brand values.
    Technology_Use: Evaluating the use of technology, apps, or online platforms.
    Return_Policy: Assessing the ease, fairness, or flexibility of return or exchange policies.
    Complaint_Handling: Rating the handling, resolution, or satisfaction of complaints.
    Marketing_Communication: Commenting on advertising, promotions, or marketing messages.
    Community_Involvement: Reflecting on the store's engagement with local community or charity.
    Environmental_Impact: Assessing the store's impact on the environment or eco-friendly practices.
"""

legal_document_focus_presets = """
    Precise: Using exact and clear language to define terms or conditions.
    Ambiguous: Containing unclear or vague language that may lead to confusion.
    Binding: Reflecting a strong commitment or obligation to adhere to the terms.
    Permissive: Allowing flexibility or granting permissions within the agreement.
    Restrictive: Imposing limitations or constraints on actions or behavior.
    Comprehensive: Covering all necessary aspects or details in a thorough manner.
    Confrontational: Taking an aggressive or adversarial stance in the language.
    Conciliatory: Seeking to smooth over differences or reach an amicable agreement.
    Formal: Adhering to conventional legal standards and language.
    Informal: Using a more casual or non-traditional tone.
    Fair: Reflecting an even-handed or equitable approach to the subject matter.
    Biased: Showing favoritism or prejudice towards a particular party or interest.
    Transparent: Providing clear and open disclosure of all relevant information.
    Confidential: Emphasizing the privacy or secrecy of the information contained.
    Enforceable: Reflecting terms that are legally binding and can be enforced.
    Unenforceable: Containing provisions that may not be legally upheld.
    Protective: Including clauses or terms to safeguard interests or rights.
    Risky: Reflecting potential legal risks or liabilities.
    Cooperative: Encouraging collaboration or mutual agreement between parties.
    Adversarial: Reflecting opposition or conflict between the interests of parties.
"""

news_article_focus_presets = """
    Objective: Presenting facts without personal emotion or bias.
    Subjective: Reflecting personal opinions, feelings, or interpretations.
    Informative: Providing valuable information, details, or insights.
    Sensational: Using exciting or shocking language to attract attention.
    Critical: Offering criticism or analysis, often pointing out flaws or problems.
    Praise: Expressing approval, admiration, or complimenting aspects of the subject.
    Neutral: Lacking strong emotions, opinions, or bias; presenting a balanced view.
    Biased: Showing favoritism or prejudice towards a particular side or perspective.
    Alarming: Conveying a sense of urgency or warning, often through dramatic language.
    Reassuring: Providing comfort or confidence, often calming concerns or fears.
    Analytical: Offering a detailed examination or analysis of the subject matter.
    Opinionated: Expressing strong personal beliefs or judgments.
    Factual: Sticking strictly to verifiable facts and data.
    Speculative: Engaging in conjecture or hypothesis without firm evidence.
    Positive: Conveying a favorable or uplifting sentiment.
    Negative: Indicating an unfavorable, critical, or discouraging sentiment.
    Cautious: Reflecting careful or guarded language, often related to uncertain matters.
    Aggressive: Using forceful or assertive language, often to make a strong point.
    Sympathetic: Showing empathy, understanding, or compassion towards a subject.
    Controversial: Addressing topics or opinions that may provoke strong disagreement.
"""

research_paper_focus_presets = """
    Insightful: Demonstrating a deep understanding or novel perspective on the subject.
    Rigorous: Reflecting strict adherence to scientific methods and principles.
    Innovative: Introducing new ideas, methods, or approaches.
    Confusing: Lacking clarity or coherence in arguments or explanations.
    Comprehensive: Providing an extensive and thorough examination of the topic.
    Biased: Showing a lack of objectivity or impartiality in presentation or analysis.
    Objective: Presenting information and arguments in a neutral and unbiased manner.
    Superficial: Lacking depth or detailed exploration of the subject.
    Relevant: Directly related or applicable to the field or subject of interest.
    Irrelevant: Containing content that does not pertain to the main subject or focus.
    Credible: Demonstrating reliability, supported by valid evidence or citations.
    Speculative: Based on conjecture or hypothesis without solid evidence.
    Well-Structured: Organized and logically structured, facilitating understanding.
    Disorganized: Lacking clear structure or logical flow, hindering comprehension.
    Convincing: Presenting compelling arguments or evidence that persuades the reader.
    Unconvincing: Failing to provide persuasive evidence or reasoning.
    Original: Offering unique contributions or fresh perspectives to the field.
    Derivative: Lacking originality, heavily relying on existing works.
    Accessible: Written in a manner that is easily understood by a broader audience.
    Technical: Utilizing specialized language or concepts that may be challenging for a general audience.
"""

blog_post_focus_presets = """
    Informative: Providing valuable insights, facts, or information.
    Personal: Sharing personal experiences, feelings, or opinions.
    Entertaining: Engaging readers with humor, stories, or amusing content.
    Inspirational: Encouraging positive feelings, creativity, or motivation.
    Technical: Delving into technical details, specifications, or processes.
    Opinionated: Expressing strong personal beliefs or judgments.
    Reflective: Encouraging thoughtful contemplation or introspection.
    Conversational: Using a casual, friendly tone to foster reader engagement.
    Professional: Maintaining a formal, business-like tone and content.
    Trendy: Covering current trends, pop culture, or hot topics.
    Educational: Offering instructional content, tutorials, or learning resources.
    Controversial: Addressing divisive or hotly debated topics.
    Encouraging: Providing support, encouragement, or positive reinforcement.
    Critical: Analyzing subjects in depth, often pointing out flaws or areas for improvement.
    Collaborative: Inviting reader interaction, comments, or collaboration.
    Visual: Relying heavily on images, videos, or visual elements to convey the message.
    Narrative: Telling a story or creating a narrative thread throughout the post.
    Promotional: Highlighting products, services, or events for marketing purposes.
    Analytical: Offering detailed analysis or examination of a particular subject.
    Concise: Providing information in a clear and brief manner, avoiding unnecessary details.
"""

book_focus_presets = """
    Engaging: Capturing the reader's interest or attention; compelling.
    Boring: Lacking excitement or interest; monotonous.
    Thoughtful: Providing deep insights or reflections; contemplative.
    Shallow: Lacking depth or complexity in theme or characterization.
    Inspirational: Offering motivation or encouragement; uplifting.
    Tragic: Conveying a sense of sorrow, loss, or despair.
    Comedic: Incorporating humor or satire; entertaining.
    Informative: Educating the reader on a particular subject or idea.
    Confusing: Leading to misunderstandings or difficulty following the plot.
    Realistic: Portraying believable characters, settings, or events.
    Fantastical: Incorporating magical or supernatural elements; imaginative.
    Suspenseful: Creating tension or anticipation; thrilling.
    Predictable: Lacking surprises or originality in plot or character development.
    Original: Offering unique or innovative concepts, themes, or storytelling.
    Emotional: Evoking strong feelings or reactions from the reader.
    Detached: Lacking emotional connection or empathy with characters.
    Complex: Featuring intricate plotlines, themes, or character relationships.
    Simplistic: Overly straightforward or lacking nuance; elementary.
    Poetic: Utilizing beautiful or artistic language; lyrical.
    Dry: Lacking liveliness or emotion; dull or academic.
"""

movie_focus_presets = """
    Artistic: Emphasizing visual aesthetics, design, or creative expression.
    Commercial: Focused on mainstream appeal, entertainment value, or box office potential.
    Realistic: Striving for authenticity, factual accuracy, or true-to-life depiction.
    Fantastical: Embracing imagination, otherworldly elements, or magical realism.
    Thought-provoking: Encouraging deep reflection, intellectual engagement, or philosophical exploration.
    Light-hearted: Aimed at amusement, fun, or easy-going entertainment.
    Dark: Delving into serious, grim, or disturbing themes and moods.
    Action-packed: Featuring intense physical activity, stunts, or thrilling sequences.
    Dialogue-driven: Relying on strong writing, character interaction, or verbal storytelling.
    Visual-driven: Utilizing striking imagery, special effects, or cinematography.
    Character-focused: Concentrating on character development, personalities, or emotional arcs.
    Plot-focused: Prioritizing a well-structured, engaging, or complex storyline.
    Original: Offering unique concepts, unconventional approaches, or innovative ideas.
    Formulaic: Following established patterns, genre conventions, or predictable structures.
    Inclusive: Promoting diversity, representation, or social awareness.
    Exclusive: Lacking in diversity or representing a narrow perspective.
    Historical: Based on or inspired by real historical events, figures, or periods.
    Futuristic: Exploring future scenarios, technological advancements, or speculative ideas.
    Inspirational: Encouraging motivation, positivity, or uplifting emotions.
    Depressing: Conveying sadness, despair, or disheartening themes.
    Balanced: Presenting multiple viewpoints, or a fair and nuanced portrayal of events.
    Biased: Favoring a particular ideology, perspective, or interpretation.
    Family-friendly: Suitable for all ages, emphasizing moral values or wholesome content.
    Mature: Containing content for adult audiences, such as violence, language, or sexuality.
    Experimental: Challenging traditional forms, embracing avant-garde techniques or ideas.
    Conventional: Adhering to well-established styles, formats, or genre expectations.
"""

tv_show_focus_presets = """
    Entertaining: Providing amusement or enjoyment through engaging content.
    Educational: Offering information or insights in an instructive manner.
    Dramatic: Emphasizing tension, conflict, or emotional intensity.
    Comedic: Focused on humor, wit, or satire to provoke laughter.
    Thrilling: Creating suspense or excitement through plot twists and action.
    Realistic: Depicting characters or scenarios in a believable and relatable way.
    Fantastical: Delving into imaginative or supernatural themes and settings.
    Family-Friendly: Suitable for all ages, often with moral lessons or wholesome themes.
    Mature: Containing content aimed at adult audiences, possibly with strong language or themes.
    Inspiring: Encouraging positive attitudes, motivation, or personal growth.
    Controversial: Tackling subjects that may provoke debate or strong opinions.
    Romantic: Focusing on love, relationships, or emotional connections.
    Action-Packed: Emphasizing physical action, stunts, or high-energy sequences.
    Reflective: Encouraging thought or introspection on deeper themes or values.
    Serialized: Following a continuous storyline across multiple episodes or seasons.
    Episodic: Each episode stands alone, with no overarching plot connecting them.
    Innovative: Breaking new ground or experimenting with unconventional storytelling.
    Traditional: Adhering to established genres or narrative conventions.
    Inclusive: Representing diverse characters or perspectives in an equitable way.
    Exclusive: Focused on a specific niche, culture, or group, possibly lacking wider diversity.
    High-Budget: Featuring high production values, with extensive resources spent on quality.
    Low-Budget: Produced with limited resources, possibly reflecting in production quality.
    Live: Broadcasted in real-time or featuring live performances.
    Scripted: Following a predetermined script or structure in the content.
    Unscripted: Allowing for improvisation, often seen in reality or talk shows.
    Celebrity-Focused: Centering on the lives, careers, or appearances of famous individuals.
    Documentary-Style: Presenting information in a factual or investigative manner.
"""

podcast_focus_presets = """
    Engaging: Capturing interest or attention through compelling content or delivery.
    Informative: Providing valuable information, insights, or expertise.
    Entertaining: Offering amusement or enjoyment through humor, storytelling, or personality.
    Monotonous: Lacking variation or excitement; potentially boring or tedious.
    Thoughtful: Reflecting careful consideration, depth, or insight.
    Superficial: Lacking depth or substance; possibly glossing over complex topics.
    Controversial: Addressing topics that may provoke disagreement or debate.
    Inspirational: Motivating or uplifting, often through personal stories or encouragement.
    Confrontational: Engaging in aggressive or challenging discussions or debates.
    Collaborative: Featuring cooperative interactions between hosts, guests, or listeners.
    Technical: Delving into specialized or complex subjects, possibly requiring background knowledge.
    Accessible: Easily understood by a general audience; avoiding jargon or complexity.
    Empathetic: Expressing understanding or compassion, especially in sensitive topics.
    Analytical: Providing a detailed examination or interpretation of a subject.
    Casual: Using a relaxed and conversational tone or approach.
    Structured: Following a clear and organized format or agenda.
    Chaotic: Lacking organization or clarity; possibly confusing or disjointed.
    Inclusive: Encouraging participation or representation of diverse perspectives.
    Biased: Reflecting a particular stance or viewpoint, possibly to the exclusion of others.
    Professional: Maintaining a formal, polished, or business-like tone or approach.
"""

song_focus_presets = """
    Emotional: Conveying strong feelings or sentiments.
    Narrative: Telling a story or depicting a series of events.
    Inspirational: Encouraging positive thoughts, motivation, or self-belief.
    Romantic: Focusing on love, affection, or intimate relationships.
    Political: Addressing political issues, activism, or social commentary.
    Party: Creating an energetic, fun, or danceable vibe.
    Reflective: Encouraging introspection, contemplation, or self-awareness.
    Experimental: Utilizing unconventional sounds or structures.
    Traditional: Adhering to classical or established musical forms.
    Modern: Embracing contemporary styles, sounds, or themes.
    Spiritual: Focusing on religious, mystical, or philosophical concepts.
    Aggressive: Utilizing intense, forceful, or confrontational language and music.
    Soothing: Creating a calming, relaxing, or tranquil atmosphere.
    Nostalgic: Recalling or celebrating the past, often with a sentimental tone.
    Optimistic: Conveying a positive, hopeful, or forward-looking perspective.
    Pessimistic: Expressing a negative, cynical, or downbeat view.
    Celebratory: Marking a special occasion, achievement, or joyful event.
    Mourning: Reflecting grief, loss, or sorrow.
    Humorous: Incorporating comedy, satire, or playful elements.
    Abstract: Emphasizing non-literal or symbolic expressions and themes.
    Personal: Detailing personal experiences, thoughts, or feelings.
    Universal: Addressing themes or ideas that resonate with a wide audience.
    Upbeat: Featuring lively, cheerful, or uplifting musical elements.
    Melancholic: Creating a mood of sadness, longing, or wistfulness.
    Empowering: Encouraging self-confidence, strength, or empowerment.
"""

video_game_focus_presets = """
    Immersive: Providing a deep and engaging experience that draws players in.
    Repetitive: Lacking variety or innovation, leading to monotonous gameplay.
    Challenging: Offering difficulties or obstacles that test players' skills.
    Casual: Designed for relaxed play without intense commitment or challenge.
    Competitive: Encouraging rivalry and competition between players.
    Cooperative: Promoting teamwork and collaboration among players.
    Story-driven: Centered around a strong narrative or plot.
    Mechanic-focused: Emphasizing gameplay mechanics and systems over story.
    Innovative: Introducing new ideas, concepts, or gameplay mechanics.
    Traditional: Sticking to well-established genres or gameplay conventions.
    Accessible: Designed to be easily understood and played by a wide audience.
    Complex: Requiring deep understanding or mastery of intricate mechanics.
    Visually-Stunning: Featuring impressive graphics, art, or visual effects.
    Auditory: Utilizing impactful sound design or music to enhance the experience.
    Ethical: Reflecting moral choices or values within the game's content.
    Violent: Containing intense or graphic violence as a core aspect.
    Educational: Offering learning opportunities or educational value.
    Social: Encouraging interaction, communication, or connection between players.
    Solo: Focused on single-player experiences without multiplayer components.
    Multiplatform: Available on various gaming platforms or devices.
    Exclusive: Restricted to a specific platform or device.
    Inclusive: Incorporating diverse characters, themes, or accessibility options.
    Microtransaction-heavy: Relying on in-game purchases for content or progression.
    Mod-friendly: Allowing or encouraging player-created modifications or content.
    Family-friendly: Suitable for players of all ages without mature content.
    Realistic: Aiming for lifelike graphics, physics, or simulation.
    Stylized: Utilizing a unique or artistic visual style distinct from realism.
"""

restaurant_review_focus_presets = """
    Delicious: Expressing enjoyment or satisfaction with the taste of the food.
    Disappointing: Indicating dissatisfaction or unmet expectations with the overall experience.
    Friendly: Reflecting warm or welcoming service from the staff.
    Unprofessional: Indicating a lack of professionalism or courtesy in service.
    Cozy: Describing a comfortable or intimate ambiance.
    Noisy: Referring to a loud or disruptive environment.
    Overpriced: Believing that the prices are too high for the value provided.
    Bargain: Finding the prices to be reasonable or a good value for the money.
    Creative: Appreciating innovative or unique dishes and presentation.
    Bland: Describing food that lacks flavor or seasoning.
    Fresh: Valuing the use of fresh ingredients and quality produce.
    Stale: Indicating that food seemed old or not freshly prepared.
    Clean: Observing cleanliness in the dining area, kitchen, or restrooms.
    Crowded: Reflecting a packed or overly busy dining space.
    Romantic: Finding the setting or atmosphere suitable for a romantic occasion.
    Family-friendly: Appreciating an environment suitable for family dining or children.
    Authentic: Valuing a genuine or traditional culinary experience.
    Fusion: Enjoying a blend or mix of culinary traditions or flavors.
    Attentive: Praising the attentive and responsive service from the staff.
    Ignored: Feeling neglected or overlooked by the service staff.
"""

hotel_review_focus_presets = """
    Comfortable: Emphasizing the comfort and coziness of the rooms and facilities.
    Uncomfortable: Highlighting discomfort or issues with room amenities.
    Luxurious: Focusing on opulence, high-end services, and lavish features.
    Budget-friendly: Stressing affordability and value for money.
    Clean: Noting the cleanliness and tidiness of the premises.
    Dirty: Pointing out hygiene issues or unclean conditions.
    Friendly: Emphasizing warmth, friendliness, or hospitality of the staff.
    Rude: Discussing impoliteness or unsatisfactory interactions with staff.
    Convenient: Highlighting the convenient location or accessibility.
    Isolated: Mentioning challenges related to location or accessibility.
    Safe: Noting the safety features and sense of security in the property.
    Unsafe: Pointing out security concerns or potential risks.
    Relaxing: Focusing on the calming ambiance, spa, or relaxation amenities.
    Noisy: Discussing noise issues or disturbances during the stay.
    Family-friendly: Emphasizing facilities and services suitable for families.
    Not-family-friendly: Lacking features or ambiance suitable for family stays.
    Pet-friendly: Noting accommodations and facilities for pets.
    Not-pet-friendly: Highlighting restrictions or inconveniences related to pets.
    Culinary: Focusing on the quality, variety, and taste of the food and beverages.
    Disappointing-food: Criticizing the food quality, taste, or presentation.
    Eco-friendly: Emphasizing sustainability practices and eco-conscious efforts.
    Not-eco-friendly: Pointing out a lack of environmental responsibility.
    Well-maintained: Noting the good condition and maintenance of the property.
    Poorly-maintained: Discussing wear and tear, or maintenance issues.
    Informative: Providing detailed information about the hotel and its services.
    Vague: Lacking clarity or detailed information about the experience.
    Romantic: Emphasizing aspects suitable for couples or romantic getaways.
    Business-oriented: Focusing on facilities and services for business travelers.
"""

travel_review_focus_presets = """
    Informative: Providing detailed information about the location, amenities, or experience.
    Misleading: Containing information that may confuse or misrepresent the actual experience.
    Enthusiastic: Expressing great excitement and positive feelings about the travel experience.
    Disappointed: Reflecting dissatisfaction or unmet expectations during the travel.
    Adventurous: Highlighting unique or thrilling aspects of the trip, such as outdoor activities.
    Relaxing: Focusing on the calming and leisurely aspects of the travel destination.
    Cultural: Emphasizing the cultural experiences, local traditions, or historical sites.
    Touristy: Focusing on popular attractions or experiences typically sought by tourists.
    Off-the-Beaten-Path: Describing lesser-known or unconventional travel experiences.
    Luxurious: Detailing high-end accommodations, gourmet dining, or exclusive services.
    Budget-Friendly: Emphasizing affordable options, deals, or value for money.
    Family-Friendly: Highlighting aspects suitable for family travel, such as child-friendly activities.
    Romantic: Capturing the intimate or romantic ambiance of a destination or accommodation.
    Eco-Friendly: Focusing on sustainability, environmental practices, or eco-tourism.
    Accessible: Detailing facilities or services catering to travelers with disabilities.
    Photogenic: Emphasizing scenic beauty or opportunities for photography.
    Gastronomic: Centering on culinary experiences, local cuisines, or dining options.
    Nightlife: Highlighting entertainment, clubs, bars, or nighttime activities.
    Safety-Conscious: Reflecting on the safety measures, local crime, or health concerns.
    Crowded: Describing the level of tourist traffic, congestion, or crowdedness.
    Seasonal: Focusing on the best time to visit or seasonal attractions and events.
    Biased: Showing favoritism towards certain aspects or a particular view of the destination.
    Balanced: Presenting a fair and comprehensive overview, including both positives and negatives.
"""


focus_presets = {
    "financial_investor_focus": financial_investor_focus_presets,
    "public_company_earnings_call_focus": public_company_earnings_call_text,
    "customer_feedback_focus": customer_feedback_focus_presets,
    "customer_service_interaction_focus": customer_service_interaction_focus_presets,
    "marketing_campaign_focus": marketing_campaign_focus_presets,
    "product_review_focus": product_review_focus_presets,
    "email_correspondence_focus": email_correspondence_focus_presets,
    "github_issues_focus": github_issues_focus_presets,
    "social_media_sentiment_focus": social_media_sentiment_focus_presets,
    "employee_feedback_focus": employee_feedback_focus_presets,
    "crisis_communication_focus": crisis_communication_focus_presets,
    "political_speech_focus": political_speech_focus_presets,
    "healthcare_patient_feedback_focus": healthcare_patient_feedback_focus_presets,
    "educational_content_focus": educational_content_focus_presets,
    "job_interview_focus": job_interview_focus_presets,
    "sales_call_focus": sales_call_focus_presets,
    "real_estate_listing_focus": real_estate_listing_focus_presets,
    "mental_health_counseling_focus": mental_health_counseling_focus_presets,
    "human_resources_recruitment_focus": human_resources_recruitment_focus_presets,
    "political_campaign_focus": political_campaign_focus_presets,
    "government_policy_response_focus": government_policy_response_focus_presets,
    "retail_customer_feedback_focus": retail_customer_feedback_focus_presets,
    "legal_document_focus": legal_document_focus_presets,
    "news_article_focus": news_article_focus_presets,
    "research_paper_focus": research_paper_focus_presets,
    "blog_post_focus": blog_post_focus_presets,
    "book_focus": book_focus_presets,
    "movie_focus": movie_focus_presets,
    "tv_show_focus": tv_show_focus_presets,
    "podcast_focus": podcast_focus_presets,
    "song_focus": song_focus_presets,
    "video_game_focus": video_game_focus_presets,
    "restaurant_review_focus": restaurant_review_focus_presets,
    "hotel_review_focus": hotel_review_focus_presets,
    "travel_review_focus": travel_review_focus_presets,
}

focus_areas_dict = {
    "financial_investor_focus": "Public markets investors trying to evaluate the comments of management teams and other professional investors and stock analysts.",
    "public_company_earnings_call_focus": "Investors, analysts, and shareholders seeking insights into a company's financial health and future prospects through earnings calls.",
    "customer_feedback_focus": "Businesses and product managers looking to understand customer satisfaction, preferences, and areas for improvement.",
    "customer_service_interaction_focus": "Customer service professionals and management aiming to evaluate and enhance customer support interactions.",
    "marketing_campaign_focus": "Marketers and business strategists interested in evaluating the effectiveness and reach of marketing campaigns.",
    "product_review_focus": "Consumers, retailers, and manufacturers seeking insights into product quality, usability, and appeal.",
    "email_correspondence_focus": "Professionals and organizations aiming to improve communication efficiency and clarity in email interactions.",
    "github_issues_focus": "Developers, project managers, and contributors looking to identify, track, and resolve issues in software development projects.",
    "social_media_sentiment_focus": "Social media analysts and brand managers monitoring public sentiment, trends, and reactions on social platforms.",
    "employee_feedback_focus": "HR professionals and management focusing on employee satisfaction, workplace culture, and opportunities for growth.",
    "crisis_communication_focus": "PR and crisis management teams analyzing and formulating responses to urgent and sensitive situations.",
    "political_speech_focus": "Political analysts, campaigners, and engaged citizens evaluating the content, strategy, and impact of political speeches.",
    "healthcare_patient_feedback_focus": "Healthcare providers and administrators seeking to understand patient experiences, concerns, and satisfaction.",
    "educational_content_focus": "Educators, students, and educational institutions assessing the quality, relevance, and effectiveness of educational materials.",
    "job_interview_focus": "Job seekers and hiring managers interested in evaluating the qualifications, experience, and fit of candidates for a specific role.",
    "sales_call_focus": "Sales professionals and potential customers assessing the value, features, and benefits of a product or service during a sales interaction.",
    "real_estate_listing_focus": "Property buyers, sellers, and real estate agents examining the details, location, and pricing of real estate properties listed for sale or rent.",
    "mental_health_counseling_focus": "Mental health professionals and individuals seeking therapy, focusing on the assessment, treatment, and support for mental and emotional well-being.",
    "human_resources_recruitment_focus": "HR professionals, recruiters, and job seekers evaluating the recruitment process, candidate sourcing, and talent acquisition strategies.",
    "political_campaign_focus": "Political strategists, voters, and media analyzing the strategies, messages, and objectives of a political campaign or candidate.",
    "government_policy_response_focus": "Policymakers, citizens, and analysts evaluating government actions, policies, and responses to specific issues or crises.",
    "retail_customer_feedback_focus": "Retail business owners, managers, and customers reviewing feedback on products, services, and overall customer experience in a retail setting.",
    "legal_document_focus": "Lawyers, legal scholars, and interested parties examining legal documents, contracts, and agreements for compliance, interpretation, and understanding.",
    "news_article_focus": "Journalists, readers, and media analysts focusing on the content, accuracy, and relevance of news articles, reports, and journalistic pieces.",
    "research_paper_focus": "Academics, researchers, students, and professionals seeking in-depth analysis, findings, methodologies, and insights in a specific field of study.",
    "blog_post_focus": "General readers, enthusiasts, or professionals interested in insights, opinions, updates, or personal narratives on specific topics, industries, or hobbies.",
    "book_focus": "Readers of various age groups and interests seeking entertainment, knowledge, or literary enrichment through novels, non-fiction, educational texts, etc.",
    "movie_focus": "Moviegoers, film enthusiasts, critics, and general audiences seeking entertainment, artistic expression, or thematic exploration through cinema.",
    "tv_show_focus": "Television viewers, fans, and critics looking for entertainment, information, or engagement with serialized stories, documentaries, talk shows, etc.",
    "podcast_focus": "Listeners interested in audio content that provides insights, entertainment, interviews, or discussions on specific subjects, industries, or trends.",
    "song_focus": "Music lovers, musicians, critics, and general audiences seeking emotional expression, entertainment, or artistic exploration through musical compositions.",
    "video_game_focus": "Gamers, game developers, critics, and enthusiasts seeking interactive entertainment, challenges, storytelling, or virtual experiences through gaming.",
    "restaurant_review_focus": "Diners, food enthusiasts, tourists, and locals looking for insights into the quality, ambiance, service, and culinary offerings of eateries and restaurants.",
    "hotel_review_focus": "Travelers, tourists, business professionals, and event planners seeking information on the comfort, amenities, location, and services of hotels and accommodations.",
    "travel_review_focus": "Travelers, adventure seekers, families, and tourists looking for insights into destinations, experiences, accommodations, attractions, and travel services."
}


def generate_all_prompts(focus_key, scoring_scale_explanation):
    preset_text = focus_presets[focus_key]
    target_audience = focus_areas_dict[focus_key]
    populated_prompts = {}
    for line in preset_text.strip().split("\n"):
        sentiment_adjective, sentiment_explanation = line.strip().split(": ", 1)
        populated_prompt_text = generate_llm_sentiment_score_prompt(sentiment_adjective, sentiment_explanation, target_audience, scoring_scale_explanation)
        populated_prompts[sentiment_adjective] = populated_prompt_text
    return populated_prompts

def generate_llm_sentiment_score_prompt(sentiment_adjective, sentiment_explanation, target_audience, scoring_scale_explanation):
    populated_prompt_text = (
        f"We are seeking to evaluate the sentiment expressed in a given text with regard to a specific adjective. "
        f"The goal is to determine the extent to which the text reflects the sentiment described by the adjective "
        f"{scoring_scale_explanation} "
        f"Focus on adjectives that would also be relevant to {target_audience}.\n\n"
        f"<sentiment_adjective>:  {sentiment_adjective}\n"
        f"<sentiment_explanation>:  {sentiment_explanation}\n\n"
        f"Please provide your analysis in the following format:\n"
        f"`sentiment_score` | `score_justification`\n"
        f"where `sentiment_score` is the score on the aforementioned scale, and `score_justification` is a textual description of what exactly made you come up with the score that you did."
    )
    return populated_prompt_text

def combine_populated_prompts_with_source_text(populated_prompt, source_text):
    combined_prompt = populated_prompt + "\n---\n" + source_text
    return combined_prompt

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self
    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )
        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )
        self.outnull_file.close()
        self.errnull_file.close()

def load_inference_model(model_name: str):
    try:
        models_dir = os.path.join(BASE_DIRECTORY, 'models') # Determine the model directory path
        matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*")) # Search for matching model files
        if not matching_files:
            logger.error(f"No model file found matching: {model_name}")
            raise FileNotFoundError
        matching_files.sort(key=os.path.getmtime, reverse=True) # Sort the files based on modification time (recently modified files first)
        model_file_path = matching_files[0]
        model_instance = Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=False, use_mlock=True) # Load the model
        return model_instance
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        raise FileNotFoundError(f"No model file found matching: {model_name}")
        
def validate_llm_generated_sentiment_response(llm_raw_output, lowest_possible_score, highest_possible_score):
    use_verbose = 1
    if use_verbose:
        logger.info(f"Validating LLM raw output: {llm_raw_output}")
    llm_raw_output = llm_raw_output.strip().strip("`")  # Remove surrounding whitespace and backticks if present
    llm_raw_output = llm_raw_output.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">") # Replace any HTML-encoded characters
    parts = llm_raw_output.split("|")
    if '|' not in llm_raw_output: # Check if delimiter '|' is present; if not, try alternative delimiters
        parts = llm_raw_output.split("_|_")
        if len(parts) < 2:
            # If no delimiter found, try extracting the first numerical value
            match = re.search(r"[-+]?\d*\.\d+|\d+", llm_raw_output)
            if match:
                sentiment_score_str = match.group(0)
                score_justification = llm_raw_output.replace(sentiment_score_str, '').strip()
            else:
                raise ValueError("Unable to extract sentiment score and justification from the LLM raw output.")
        else:
            sentiment_score_str, score_justification = parts[0], "|".join(parts[1:])
    else:
        sentiment_score_str, score_justification = parts[0], "|".join(parts[1:])
        logger.info(f"Extracted sentiment_score_str: {sentiment_score_str}")
        logger.info(f"Extracted score_justification: {score_justification}")
    sentiment_score_str = sentiment_score_str.strip().rstrip("`")  # Trim any leading or trailing whitespace from both parts and trailing backtick if present
    score_justification = score_justification.strip().rstrip("`") 
    if use_verbose:
        logger.info(f"Trimmed sentiment_score_str: {sentiment_score_str}")
    sentiment_score_str = "".join(char for char in sentiment_score_str if char in "0123456789.-") # Remove any non-numeric characters except the decimal point and negative sign from sentiment_score_str
    if use_verbose:
        logger.info(f"Removed non-numeric characters from sentiment_score_str: {sentiment_score_str}")
    if sentiment_score_str.count(".") > 1: # If multiple decimal points or negative signs are present, keep only the first occurrence
        sentiment_score_str = sentiment_score_str.replace(".", "", sentiment_score_str.count(".") - 1)
    if sentiment_score_str.count("-") > 1:
        sentiment_score_str = sentiment_score_str.replace("-", "", sentiment_score_str.count("-") - 1)
    if use_verbose:
        logger.info(f"Removed extra decimal points and negative signs from sentiment_score_str: {sentiment_score_str}")
    try: # Attempt to cast sentiment_score into a float
        if use_verbose:
            logger.info(f"Attempting to cast sentiment_score_str {sentiment_score_str} into a float...")
        sentiment_score = float(sentiment_score_str)
        if sentiment_score < lowest_possible_score: # If out of range, bound to the nearest limit
            logger.warning(f"Sentiment score {sentiment_score} is below the lower bound {lowest_possible_score}.")
            sentiment_score = lowest_possible_score
        elif sentiment_score > highest_possible_score:
            logger.warning(f"Sentiment score {sentiment_score} is above the upper bound {highest_possible_score}.")
            sentiment_score = highest_possible_score
    except ValueError:
        logger.error(f"Sentiment score {sentiment_score_str} could not be cast into a float after attempted corrections.")
        raise ValueError("Sentiment score could not be cast into a float after attempted corrections.")
    if len(score_justification) < 30 or len(score_justification.split()) < 5: # If score justification is too short, return a warning in the justification itself
        logger.warning(f"Justification is too short: {score_justification}")
        score_justification = f"Warning: Justification is too short. Original response: {score_justification}"
    return sentiment_score, score_justification

def run_llm_in_process(combined_prompt_text, model_name):
    try:
        logger.info("Running run_llm_in_process with model_name: " + model_name)
        logger.info("Combined prompt text: " + combined_prompt_text[:100])  # Log the first 100 characters
        with suppress_stdout_stderr():
            logger.info("Loading LLM model...")
            llm_local = load_inference_model(model_name)
            logger.info("Model loaded successfully.")
        logger.info("Running model with combined prompt text...")
        result = llm_local(combined_prompt_text)
        logger.info("Model run successful.")
        return result
    except Exception as e:
        traceback.print_exc()
        logger.error(f"An error occurred in run_llm_in_process: {e}", exc_info=True, stack_info=True)

async def parallel_attempt(combined_prompt_text, model_name):
    global lowest_possible_score, highest_possible_score
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        result_dict = await loop.run_in_executor(executor, partial(run_llm_in_process, combined_prompt_text, model_name))
        result_text = result_dict['choices'][0]['text']  # Extract the text from the dictionary
        return result_text  # Returning raw result text

def combine_llm_generated_sentiment_responses(llm_raw_outputs, lowest_possible_score, highest_possible_score):
    scores = []
    score_justification_strings = []
    failed_runs = 0
    min_successful_runs = 3 # Set as appropriate
    for llm_raw_output in llm_raw_outputs:
        try:
            sentiment_score, score_justification = validate_llm_generated_sentiment_response(llm_raw_output, lowest_possible_score, highest_possible_score)
            scores.append(sentiment_score)
            score_justification_strings.append(score_justification)
        except ValueError:
            failed_runs += 1
    if len(scores) < min_successful_runs: # Fallback strategy, e.g., return an error, use a default value, etc.
        raise ValueError("Insufficient successful parallel runs.")
    mean_sentiment_score = sum(scores) / len(scores)
    bootstrap_samples = [np.random.choice(scores, len(scores)) for _ in range(1000)] # Calculate the 95% confidence interval using the bootstrap method
    sentiment_score_95_pct_confidence_interval = np.percentile(bootstrap_samples, [2.5, 97.5])
    interquartile_range_of_sentiment_scores = np.percentile(bootstrap_samples, [25, 75])
    iqr_of_sentiment_score_as_pct_of_mean_score = (interquartile_range_of_sentiment_scores[1] - interquartile_range_of_sentiment_scores[0]) / mean_sentiment_score
    return mean_sentiment_score, sentiment_score_95_pct_confidence_interval, interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score, score_justification_strings

def update_summary_table(new_row):
    lock = FileLock(SUMMARY_TABLE_PATH + ".lock")
    with lock:
        summary_table = pd.read_csv(SUMMARY_TABLE_PATH) # Read the existing summary table
        summary_table.loc[len(summary_table)] = new_row # Add the new row
        summary_table.to_csv(SUMMARY_TABLE_PATH, index=False) # Write back to the file

async def get_sentiment_score_from_llm(focus_key, sentiment_adjective, sentiment_explanation, target_audience, scoring_scale_explanation, source_text, model_name):
    start_time = datetime.utcnow()
    summary_table = pd.DataFrame(columns=['Attempt', 'Successful Runs', 'Failed Runs', 'Time Taken in Seconds', 'Mean Score', '95% CI Lower', '95% CI Upper', 'IQR Lower', 'IQR Upper', 'IQR as Pct of Mean'])
    populated_prompt_text = generate_llm_sentiment_score_prompt(sentiment_adjective, sentiment_explanation, target_audience, scoring_scale_explanation)
    combined_prompt_text = combine_populated_prompts_with_source_text(populated_prompt_text, source_text)
    backoff_time = 1
    raw_outputs = []
    for attempt in range(MAX_ATTEMPTS // PARALLEL_ATTEMPTS):
        logger.info(f"Starting attempt {attempt + 1} out of {MAX_ATTEMPTS // PARALLEL_ATTEMPTS}.")
        tasks = [parallel_attempt(combined_prompt_text, model_name) for _ in range(PARALLEL_ATTEMPTS)]
        results = await asyncio.gather(*tasks)
        raw_outputs.extend(results)
        successful_runs = 0
        failed_runs = 0
        for llm_raw_output in raw_outputs:
            try:
                validate_llm_generated_sentiment_response(llm_raw_output, lowest_possible_score, highest_possible_score)
                print()
                successful_runs += 1
            except ValueError:
                failed_runs += 1
        logger.info(f"Attempt {attempt + 1}: {successful_runs} successful runs, {failed_runs} failed runs.")
        if successful_runs >= MIN_SUCCESSFUL_RUNS_TO_GENERATE_SENTIMENT_SCORE: # Check if enough valid results have been gathered
            try:
                mean_sentiment_score, sentiment_score_95_pct_confidence_interval, interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score, score_justification_strings = combine_llm_generated_sentiment_responses(raw_outputs, lowest_possible_score, highest_possible_score)
                logger.info(f"Output validation successful! Mean Sentiment Score: {mean_sentiment_score} | Confidence Interval: {sentiment_score_95_pct_confidence_interval}")
                logger.info(f"Interquartile Range of Sentiment Scores: {interquartile_range_of_sentiment_scores} | IQR as Percentage of Mean Score: {iqr_of_sentiment_score_as_pct_of_mean_score * 100}%")
                summary_table.loc[len(summary_table)] = [attempt + 1, successful_runs, failed_runs, mean_sentiment_score, *sentiment_score_95_pct_confidence_interval, *interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score]
                finish_time = datetime.utcnow()
                time_taken_in_seconds = (finish_time - start_time).total_seconds()
                new_row = [focus_key, sentiment_adjective, combined_prompt_text, attempt + 1, successful_runs, failed_runs, time_taken_in_seconds, mean_sentiment_score, *sentiment_score_95_pct_confidence_interval, *interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score]
                update_summary_table(new_row)
                return mean_sentiment_score, sentiment_score_95_pct_confidence_interval, interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score, score_justification_strings, summary_table
            except ValueError:
                logger.error(f"Attempt {attempt + 1} failed to combine outputs despite having enough successful runs. Trying again.")
                summary_table.loc[len(summary_table)] = [attempt + 1, successful_runs, failed_runs, None, None, None, None, None, None]
        else:
            logger.warning(f"Not enough successful runs in attempt {attempt + 1}. Trying again.")
            summary_table.loc[len(summary_table)] = [attempt + 1, successful_runs, failed_runs, None, None, None, None, None, None]
        await asyncio.sleep(backoff_time) # Async sleep
        backoff_time *= 2 # Double the backoff time for the next iteration
    logger.error("Maximum attempts reached without valid output. Please review the LLM's responses.")
    raise Exception("Maximum attempts reached without valid output. Please review the LLM's responses.")

async def analyze_focus_area_sentiments(focus_key, scoring_scale_explanation, source_text, model_name):
    analysis_start_time = datetime.utcnow()
    populated_prompts_dict = generate_all_prompts(focus_key, scoring_scale_explanation)
    target_audience = focus_areas_dict[focus_key]
    combined_sentiment_analysis_dict = {
        "focus_area": focus_key,
        "target_audience": target_audience,
        "scoring_scale_explanation": scoring_scale_explanation,
        "source_text": source_text,
        "model_name": model_name,
        "lowest_possible_score": lowest_possible_score,
        "highest_possible_score": highest_possible_score,
        "individual_sentiment_report_dict": {}
    }
    for sentiment_adjective, populated_prompt_text in populated_prompts_dict.items():
        sentiment_explanation = populated_prompt_text.split("<sentiment_explanation>:  ")[1].split("\n")[0]
        mean_sentiment_score, sentiment_score_95_pct_confidence_interval, interquartile_range_of_sentiment_scores, iqr_of_sentiment_score_as_pct_of_mean_score, score_justification_strings, summary_table = await get_sentiment_score_from_llm(focus_key, sentiment_adjective, sentiment_explanation, target_audience, scoring_scale_explanation, source_text, model_name)
        combined_sentiment_analysis_dict["individual_sentiment_report_dict"][sentiment_adjective] = {
            "sentiment_adjective": sentiment_adjective,
            "sentiment_explanation": sentiment_explanation,
            "populated_prompt_text": populated_prompt_text,
            "sentiment_scores_dict": {"mean_sentiment_score": mean_sentiment_score,
                            "sentiment_score_95_pct_confidence_interval": sentiment_score_95_pct_confidence_interval,
                            "interquartile_range_of_sentiment_scores": interquartile_range_of_sentiment_scores,
                            "iqr_of_sentiment_score_as_pct_of_mean_score": iqr_of_sentiment_score_as_pct_of_mean_score,
                            "score_justification_strings": score_justification_strings,
                            "run_summary_table": summary_table.to_dict(orient="records")
            }
        }
    analysis_finish_time = datetime.utcnow()
    analysis_time_taken_in_seconds = (analysis_finish_time - analysis_start_time).total_seconds()
    logger.info(f"Analysis of {focus_key} took {analysis_time_taken_in_seconds} seconds.")
    return combined_sentiment_analysis_dict

def calculate_max_workers(model_memory_requirement, safety_factor=0.5):
    available_memory = psutil.virtual_memory().available / (1024 ** 2) # Convert to MB
    max_workers = int((available_memory * safety_factor) / model_memory_requirement)
    return max_workers

def get_model_memory_requirement(model_name):
    model_name_pattern = f"models/{model_name}*.bin"
    model_files = glob.glob(model_name_pattern)
    if model_files:
        model_file_path = model_files[0] # Assuming there's only one match
        model_memory_requirement = os.path.getsize(model_file_path) / (1024 ** 2) # Convert to MB
        return model_memory_requirement
    else:
        raise FileNotFoundError(f"No model file found matching pattern: {model_name_pattern}")


# Example usage:
lowest_possible_score = -100.0
highest_possible_score = 100.0
model_name = "wizardlm-1.0-uncensored-llama2-13b"
source_text = "The food was delicious but the service was slow." # Example source text
focus_key = "restaurant_review_focus"


model_memory_requirement = get_model_memory_requirement(model_name)
print(f"Model memory requirement: {model_memory_requirement} MB")
MAX_WORKERS = calculate_max_workers(model_memory_requirement)
print(f"MAX_WORKERS: {MAX_WORKERS}")
MAX_ATTEMPTS = 12 # How many times to attempt to generate a valid output before giving up
PARALLEL_ATTEMPTS = 3 # How many parallel attempts to make per iteration
MIN_SUCCESSFUL_RUNS_TO_GENERATE_SENTIMENT_SCORE = 5 # How many successful runs are required to consider the output valid
USE_RAMDISK = False
SUMMARY_TABLE_PATH = 'combined_summary_table.csv'
LLM_CONTEXT_SIZE_IN_TOKENS = 1024
neutral_score = lowest_possible_score + (highest_possible_score - lowest_possible_score) / 2
scoring_scale_explanation = f"on a scale from {lowest_possible_score} (strongly does NOT exhibit the adjective) to +{highest_possible_score} (strongly exhibits the adjective)-- so that {neutral_score} implies that nothing can be determined about the extent to which the adjective is reflected-- based on the contents of a sentence/paragraph/utterance."
logger.info(f"Scoring scale explanation: {scoring_scale_explanation}")
# Initialize the summary table
summary_columns = ['Focus Area', 'Sentiment Adjective', 'Prompt', 'Attempt', 'Successful Runs', 'Failed Runs', 'Mean Score', '95% CI Lower', '95% CI Upper']
initial_summary_table = pd.DataFrame(columns=summary_columns)
initial_summary_table.to_csv(SUMMARY_TABLE_PATH, index=False)

combined_sentiment_analysis_dict = asyncio.run(analyze_focus_area_sentiments(focus_key, scoring_scale_explanation, source_text, model_name))  # noqa: E501
with open('combined_sentiment_analysis.json', 'w') as file:
    json.dump(combined_sentiment_analysis_dict, file, indent=4)
print(combined_sentiment_analysis_dict)
