"""Strict explicit-role extraction; same policy as the existing customer diagnostic."""
import re

MARKER = re.compile(r"\b(Agent|Customer)\s*:", re.I)
LINE_MARKER = re.compile(r"^\s*(Agent|Customer)\s*:", re.I)

def extract_customer(text):
    """Accept explicitly labelled turns, including whitespace-separated inline turns.

    Remove marker/delimiter whitespace, keep utterance wording/internal spacing,
    and join customer turns with newline. Unlabelled nonblank lines, markers
    attached to another word, or empty turns invalidate the call; never repair.
    """
    customer, issues = [], []
    agent_count = 0
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        start = LINE_MARKER.match(line)
        # Reject attached role-like markers rather than attributing their text to
        # the previous speaker. This is stricter than word-boundary matching.
        attached = [m for m in re.finditer(r"(Agent|Customer)\s*:", line, re.I)
                    if m.start() and not line[m.start()-1].isspace()]
        if attached:
            issues.append(f"line {number}: marker attached to preceding non-whitespace text")
            continue
        markers = list(MARKER.finditer(line))
        if not start:
            issues.append(f"line {number}: nonblank text lacks explicit line-start marker")
            continue
        if any(m.start() and not line[m.start() - 1].isspace() for m in markers):
            issues.append(f"line {number}: marker attached to preceding non-whitespace text")
            continue
        for i, current in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(line)
            speech = line[current.end():end].strip()
            if not speech:
                issues.append(f"line {number}: empty attributed turn")
                continue
            if current.group(1).lower() == "customer":
                customer.append(speech)
            else:
                agent_count += 1
    if not customer:
        issues.append("no nonempty explicitly attributed customer turns")
    return {
        "valid": not issues, "issues": issues,
        "customer_transcript": "\n".join(customer) if not issues else "",
        "customer_turns_recognized": len(customer), "agent_turns_recognized": agent_count,
    }
