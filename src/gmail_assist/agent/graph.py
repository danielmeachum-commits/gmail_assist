"""The email triage graph.

    START ─► tag_me ─► route_by_kind ─┬─ personal ─► detect_login_code ─┬─ login code ─► END
                                      │                                 └─ otherwise ──► check_thread
                                      └─ bulk ─────► review_subscription ──────────────► check_thread

    check_thread ─┬─ thread already labelled ─► END (inherits it)
                  └─ new thread ──────────────► check_sender

    check_sender ─┬─ automated, sender always labelled the same ─► END (that label)
                  └─ otherwise ──────────────────────────────────► choose_model

    choose_model ─► classify ─► after_classify ─┬─ confident ──────────► decide ──────► END
       (local)         ▲                        ├─ confident, parent ──► refine_parent ► END
                       │                        ├─ unsure, local only ─► escalate ─┐
                       │                        └─ unsure, strong ─────► flag_review ─► END
                       └───────────────────────────────────────────────────────────┘

    refine_parent picks an existing sub-label, or files the message under a
    proposed one (labelled with the parent until it's approved), or keeps it.
"""
from langgraph.graph import END, START, StateGraph

from gmail_assist.agent import nodes
from gmail_assist.agent.state import Context, EmailState


def build_graph():
    builder = StateGraph(EmailState, context_schema=Context)

    builder.add_node('tag_me', nodes.tag_me)
    builder.add_node('detect_login_code', nodes.detect_login_code)
    builder.add_node('review_subscription', nodes.review_subscription)
    builder.add_node('check_thread', nodes.check_thread)
    builder.add_node('check_sender', nodes.check_sender)
    builder.add_node('choose_model', nodes.choose_model)
    builder.add_node('classify', nodes.classify)
    builder.add_node('escalate', nodes.escalate)
    builder.add_node('decide', nodes.decide)
    builder.add_node('refine_parent', nodes.refine_parent)
    builder.add_node('flag_review', nodes.flag_review)

    builder.add_edge(START, 'tag_me')
    builder.add_conditional_edges('tag_me', nodes.route_by_kind)
    builder.add_conditional_edges('detect_login_code', nodes.after_login_check)
    builder.add_edge('review_subscription', 'check_thread')
    builder.add_conditional_edges('check_thread', nodes.after_thread_check)
    builder.add_conditional_edges('check_sender', nodes.after_sender_check)
    builder.add_edge('choose_model', 'classify')
    builder.add_conditional_edges('classify', nodes.after_classify)
    builder.add_edge('escalate', 'classify')
    builder.add_edge('decide', END)
    builder.add_edge('refine_parent', END)
    builder.add_edge('flag_review', END)

    return builder.compile()
