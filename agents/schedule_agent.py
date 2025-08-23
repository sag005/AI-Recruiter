import subprocess
import re
from datetime import datetime, timedelta
from typing import List, Tuple
from crewai import Agent, Task, Crew
from crewai_tools import tool


@tool("calendar_events_tool")
def get_calendar_events() -> str:
    """Get today's calendar events using icalBuddy command"""
    try:
        result = subprocess.run(
            ['icalBuddy', '-f', '-nc', 'eventsToday'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running icalBuddy: {e}"
    except FileNotFoundError:
        return "icalBuddy not found. Please install it first."


@tool("find_available_slots_tool")
def find_available_slots(events_text: str) -> str:
    """Parse calendar events and find available 30-minute slots"""

    def parse_time(time_str: str) -> datetime:
        """Parse time string from icalBuddy output"""
        # Handle various time formats from icalBuddy
        time_formats = [
            "%I:%M %p",  # 2:30 PM
            "%H:%M",  # 14:30
            "%I:%M%p",  # 2:30PM (no space)
        ]

        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt)
                # Set today's date
                today = datetime.now().date()
                return datetime.combine(today, parsed_time.time())
            except ValueError:
                continue

        # If no format matches, try to extract numbers
        time_match = re.search(r'(\d{1,2}):(\d{2})', time_str)
        if time_match:
            hour, minute = int(time_match.group(1)), int(time_match.group(2))
            if 'PM' in time_str.upper() and hour != 12:
                hour += 12
            elif 'AM' in time_str.upper() and hour == 12:
                hour = 0
            today = datetime.now().date()
            return datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))

        return None

    def extract_events(events_text: str) -> List[Tuple[datetime, datetime]]:
        """Extract event times from icalBuddy output"""
        events = []
        lines = events_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith('•') or 'No events found' in line:
                continue

            # Look for time patterns in the line
            # Common icalBuddy formats: "Event Title (2:30 PM - 3:30 PM)"
            time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)\s*-\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)'
            match = re.search(time_pattern, line, re.IGNORECASE)

            if match:
                start_time_str = match.group(1)
                end_time_str = match.group(2)

                start_time = parse_time(start_time_str)
                end_time = parse_time(end_time_str)

                if start_time and end_time:
                    # Handle case where end time is earlier than start (next day)
                    if end_time <= start_time:
                        end_time += timedelta(days=1)
                    events.append((start_time, end_time))

        return sorted(events, key=lambda x: x[0])

    def find_gaps(events: List[Tuple[datetime, datetime]], work_start: int = 9, work_end: int = 18) -> List[
        Tuple[datetime, datetime]]:
        """Find gaps between events during work hours"""
        gaps = []
        today = datetime.now().date()
        work_start_time = datetime.combine(today, datetime.min.time().replace(hour=work_start))
        work_end_time = datetime.combine(today, datetime.min.time().replace(hour=work_end))

        if not events:
            gaps.append((work_start_time, work_end_time))
            return gaps

        # Gap before first event
        first_event_start = events[0][0]
        if first_event_start > work_start_time:
            gaps.append((work_start_time, first_event_start))

        # Gaps between events
        for i in range(len(events) - 1):
            current_end = events[i][1]
            next_start = events[i + 1][0]

            if next_start > current_end:
                gaps.append((current_end, next_start))

        # Gap after last event
        last_event_end = events[-1][1]
        if last_event_end < work_end_time:
            gaps.append((last_event_end, work_end_time))

        return gaps

    def find_30_min_slots(gaps: List[Tuple[datetime, datetime]]) -> List[str]:
        """Find 30-minute slots within gaps"""
        slots = []

        for gap_start, gap_end in gaps:
            current_time = gap_start

            while current_time + timedelta(minutes=30) <= gap_end:
                slot_end = current_time + timedelta(minutes=30)
                slot_str = f"{current_time.strftime('%I:%M %p')} - {slot_end.strftime('%I:%M %p')}"
                slots.append(slot_str)
                current_time += timedelta(minutes=15)  # 15-minute intervals for more options

        return slots[:3]  # Return earliest 3 slots

    # Parse events and find available slots
    events = extract_events(events_text)
    gaps = find_gaps(events)
    available_slots = find_30_min_slots(gaps)

    result = f"Parsed {len(events)} events from calendar.\n"
    result += f"Found {len(gaps)} time gaps.\n"
    result += f"Available 30-minute slots (earliest 3):\n"

    if available_slots:
        for i, slot in enumerate(available_slots, 1):
            result += f"{i}. {slot}\n"
    else:
        result += "No available 30-minute slots found during work hours (9 AM - 6 PM).\n"

    return result


# Create the Calendar Agent
calendar_agent = Agent(
    role='Calendar Assistant',
    goal='Find available 30-minute time slots in today\'s calendar',
    backstory='''You are a helpful calendar assistant that analyzes calendar events 
    and identifies available time slots for scheduling meetings or tasks.''',
    verbose=True,
    allow_delegation=False,
    tools=[get_calendar_events, find_available_slots_tool]
)

# Create the task
calendar_task = Task(
    description='''
    1. Use icalBuddy to get today's calendar events
    2. Parse the events to understand scheduled times
    3. Find available 30-minute time slots
    4. Return the earliest 3 available slots
    ''',
    agent=calendar_agent,
    expected_output='A list of the 3 earliest available 30-minute time slots for today'
)


# Create and run the crew
def run_calendar_crew():
    """Run the calendar crew to find available time slots"""
    crew = Crew(
        agents=[calendar_agent],
        tasks=[calendar_task],
        verbose=True
    )

    result = crew.kickoff()
    return result


# Example usage
if __name__ == "__main__":
    print("Finding available calendar slots...")
    result = run_calendar_crew()
    print("\n" + "=" * 50)
    print("RESULT:")
    print("=" * 50)
    print(result)


# Alternative direct function call (without CrewAI)
def get_available_slots_direct():
    """Direct function call without CrewAI framework"""
    print("Getting calendar events...")
    events_text = get_calendar_events()
    print(f"Events found:\n{events_text}\n")

    print("Finding available slots...")
    slots = find_available_slots_tool(events_text)
    return slots

# Uncomment to run direct version
# if __name__ == "__main__":
#     result = get_available_slots_direct()
#     print(result)