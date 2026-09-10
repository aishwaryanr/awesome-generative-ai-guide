# Forward deployed engineering: the cheat sheet

The companion cheat sheet for [Forward Deployed Engineer: Hype, Reality & a Realistic Roadmap](forward-deployed-engineering.md).

Every number here comes from 1000+ live job listings, read directly from the
companies' own job boards. Browse and filter all of them at
**[fde.levelup-labs.ai](https://fde.levelup-labs.ai)**.

---

## The roadmap

3 layers, bottom up. The [skill map on the board](https://fde.levelup-labs.ai) shows how
often the listings name each skill, and jumps you to the roles asking for it.

### Layer 1: software engineering breadth

The layer people want to skip. You can't.

| Skill | Why it matters in this role |
|---|---|
| **One language, properly** | Python or TypeScript. Python because the AI tooling is written in it, so that's where the libraries and examples live. TypeScript because a lot of what you build is something a person clicks on. |
| **APIs** | Both sides. Consuming someone else's and designing your own. Most of this job is connecting your system to theirs. |
| **Databases and SQL** | Where the customer's data lives. In a normal role you design your own schema. Here you're reading someone else's on day 1. |
| **Containers and Docker** | Packaging your application with everything it needs, so it behaves the same on their infrastructure as on your laptop. Matters more here because you deploy into environments you don't control. |
| **One cloud, plus CI and deployment** | How code gets from your machine to something running, safely and repeatably. |
| **Monitoring and logging** | So you know something has broken before the customer calls to tell you. |

The goal for the whole layer: take an application from your laptop to something real
people use, then keep it alive.

### Layer 2: AI engineering breadth

Building against a non-deterministic API. Send the same thing twice, get 2 different
answers. Everything in this layer exists to deal with that.

| Skill | Why it matters in this role |
|---|---|
| **Prompting and context engineering** | Prompting is how you ask. Context engineering is the bigger piece: deciding what goes into the model on any given call and what stays out. The context is the customer's, so getting the right slice of their documents, data and tools in front of the model is close to the whole job. |
| **Retrieval and RAG** | How the system finds the right information before answering, so you build on a company's own knowledge instead of general knowledge. |
| **Agents and tool use** | Giving the model the ability to do things: call an API, query a database, run a step in a workflow. Most of what you build in these roles is a version of this. |
| **Evaluation** | How you know the thing works. The part most people skip. Here you're usually the one deciding what good even looks like for this particular customer. |

### Layer 3: the forward deployed layer

Mostly about how you scope.

- **Scoping.** Sitting with someone and finding the real problem underneath the one they described.
- **Adoption.** Driving usage once the thing works.
- **Documentation.** Writing down what you learn so the next person doesn't start from scratch.

There's nothing you can go and read for this one. Find a real problem and build for it. It
doesn't have to be impressive, a shop down the road with a scheduling problem works. Talk
to them, work out what's actually wrong, build it, get them using it, hand it over, and
watch what happens over the next 2 months. That one exercise teaches scoping, system
design, adoption and handover at once, and you can do it before anybody hires you.

---

## Reading a listing: building job or selling job

Some companies are relabelling a solutions role. Read the posting for sales cycle
language, and for a quota attached to your name. Across forward deployed postings, about
20% mention supporting the sales cycle. For solutions engineer roles it's about 63%.

The same test applied to the neighbouring titles:

| Title | Sales language | Undefined problems | Verdict |
|---|---|---|---|
| Forward deployed engineer | 12% | 43% | reference |
| Deployment strategist | 10% | 44% | same job |
| Deployment engineer | 11% | 51% | same job |
| Implementation engineer | 43% | 38% | different job |
| Solutions architect | 46% | 18% | different job |
| Field engineer | 50% | 18% | different job |
| Solutions engineer | 57% | 13% | different job |

Low on sales language and high on undefined problems means it's the building job.

---

## The 4 myths

**"It's the highest paying job in AI."** Among US listings that disclose a band, forward
deployed posts roughly $150,000 to $210,000, against roughly $198,000 to $292,000 for
ordinary AI engineering roles at the same kind of companies. The million dollar packages
are real, and they're principal level at frontier labs, where AI engineers and researchers
are paid in that range too. That number describes those companies rather than this role.

**"You need a whole new stack."** Software engineering breadth, plus AI engineering
breadth, plus the customer work. The stack barely moves, apart from having to understand
someone else's environment.

**"It's a rebranded solutions consultant role."** Sometimes. Use the sales cycle test above
on the specific listing.

**"It's only for senior people."** Median stated minimum is 4 years, so it isn't a first
job. But 40.1% of listings state no experience requirement at all. If you've done software
engineering and then picked up AI engineering, that's close to exactly what employers are
asking for.

---

## Is the role for you

What the week looks like:

1. Meetings with people who don't work at your company
2. Sitting with someone's team working out what they actually need, which is usually different from what they asked for
3. Writing code in an environment you don't control, with tools you didn't choose
4. Explaining technical decisions to executives in language they can act on
5. Running enablement sessions so someone else's team uses what you built
6. Writing things down so the next person doesn't start from scratch
7. Travel, named in about 41% of listings

If what makes you happy is going deep on 1 hard technical problem for 6 months, this isn't
your role, and that's a reasonable thing to know about yourself.

When we hire forward deployed people, one of the biggest things we look at is
communication. Whether someone can hold a room and bring it with them.

---

## Go apply

[**fde.levelup-labs.ai**](https://fde.levelup-labs.ai) has every role collected here.
Filter by location, by experience asked for, and by the skill signals in the listing, so
you can see which are a fit before spending time on an application.
