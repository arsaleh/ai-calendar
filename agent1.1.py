import os
import json
import logging
from datetime import datetime, timedelta
import re
import tzlocal

from openai import OpenAI
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Set up Google Calendar API
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_secrets_file('credentials.json', SCOPES)
            flow.run_local_server(port=0)
            creds = flow.credentials

        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

class ContinuousChatCalendarAssistant:
    def __init__(self):
        self.conversation_history = []
        creds = get_credentials()
        self.service = build('calendar', 'v3', credentials=creds)

    def chat(self, user_message):
        llm_response = self._chat_with_llm(user_message)
        print(f"Assistant: {llm_response}")
        
        events = self._extract_calendar_events(llm_response)
        
        scheduling_results = []
        if events:
            print(f"\nDetected {len(events)} event(s) to schedule:")
            for event in events:
                print(f"- {event.get('summary', 'Untitled Event')} at {event.get('start', {}).get('dateTime', 'No time specified')}")
            
            confirmation = self._ask_for_confirmation(events)
            if confirmation.lower() == 'yes':
                scheduling_results = self._create_calendar_events(events)
            else:
                scheduling_results = ["Event creation cancelled."]
        else:
            print("No events detected for scheduling.")
        
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": llm_response})
        
        return llm_response, scheduling_results

    def _chat_with_llm(self, user_message):
        messages = self.conversation_history + [{"role": "user", "content": user_message}]
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in chat with LLM: {e}")
            return "I'm sorry, but I'm having trouble processing your request right now. Can you please try again?"

    def _extract_calendar_events(self, text):
        # Get the local timezone
        local_timezone = tzlocal.get_localzone()

        # Get the current time in the local timezone
        current_time = datetime.now(local_timezone)
        formatted_current_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

        prompt = f"""
        Based on the following text and conversation history, extract the necessary information for scheduling events and generate Google Calendar API calls in JSON format. 

        Text: {text}

        Conversation history: {self.conversation_history}

        Current date and time: {formatted_current_time}
        Local timezone: {local_timezone}

        For each event, generate a JSON object with the following structure:
        {{
           "summary": "Event title",
           "start": {{"dateTime": "YYYY-MM-DDTHH:MM:SS", "timeZone": "{local_timezone}"}},
           "end": {{"dateTime": "YYYY-MM-DDTHH:MM:SS", "timeZone": "{local_timezone}"}},
           "description": "Event description"
        }}

        Do not include a 'calendarId' field in the JSON object.
        If multiple events are mentioned, generate multiple JSON objects in a list.
        If no event scheduling is detected, return an empty list.

        Ensure your response is valid JSON. Do not include any explanatory text or code block formatting.
        Use the provided current date and time as reference for any relative dates or times (e.g., "today", "tomorrow", "next Friday", "in 2 hours").
        Always use the local timezone provided.
        Adjust all dates and times to be in the future relative to the current date and time provided.
        For any ambiguous times (e.g., "3 PM" without a date), assume it's for the next available time after the current date and time.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that extracts calendar event information and generates Google Calendar API calls."},
                    {"role": "user", "content": prompt}
                ]
            )

            llm_response = response.choices[0].message.content
            logging.debug(f"LLM response: {llm_response}")

            json_str = re.sub(r'```json\s*|\s*```', '', llm_response).strip()

            try:
                events = json.loads(json_str)
                if isinstance(events, list):
                    return self._adjust_event_dates(events)
                elif isinstance(events, dict):
                    return self._adjust_event_dates([events])
                else:
                    logging.error(f"Unexpected JSON structure: {events}")
                    return []
            except json.JSONDecodeError as json_error:
                logging.error(f"Failed to parse LLM response as JSON: {json_error}")
                logging.error(f"Cleaned JSON content: {json_str}")
                return []
        except Exception as e:
            logging.error(f"Error in extracting calendar events: {e}")
            return []

    def _adjust_event_dates(self, events):
        local_tz = tzlocal.get_localzone()
        now = datetime.now(local_tz)
        for event in events:
            start_time = datetime.fromisoformat(event['start']['dateTime'])
            end_time = datetime.fromisoformat(event['end']['dateTime'])
            
            # If datetime is naive, make it aware. If it's already aware, convert it to the local timezone.
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=local_tz)
            else:
                start_time = start_time.astimezone(local_tz)
            
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=local_tz)
            else:
                end_time = end_time.astimezone(local_tz)
            
            # If the event is in the past, adjust it to the next occurrence
            while start_time < now:
                if 'friday' in event['summary'].lower():
                    # Adjust to next Friday
                    days_ahead = 4 - start_time.weekday()
                    if days_ahead <= 0:
                        days_ahead += 7
                    start_time += timedelta(days=days_ahead)
                    end_time += timedelta(days=days_ahead)
                else:
                    # For other cases, just move it to tomorrow
                    start_time += timedelta(days=1)
                    end_time += timedelta(days=1)
            
            event['start']['dateTime'] = start_time.isoformat()
            event['end']['dateTime'] = end_time.isoformat()
        
        return events

    def _ask_for_confirmation(self, events):
        print("\nThe following events will be created:")
        for event in events:
            print(f"- {event['summary']} on {event['start']['dateTime']} for {self._calculate_duration(event)}")
        return input("Do you want to create these events? (yes/no): ")

    def _calculate_duration(self, event):
        start = datetime.fromisoformat(event['start']['dateTime'])
        end = datetime.fromisoformat(event['end']['dateTime'])
        duration = end - start
        hours, remainder = divmod(duration.seconds, 3600)
        minutes = remainder // 60
        return f"{hours} hours and {minutes} minutes"

    def _create_calendar_events(self, events):
        results = []
        for event in events:
            try:
                calendar_id = 'primary'
                logging.debug(f"Attempting to create event in calendar '{calendar_id}': {event}")
                created_event = self.service.events().insert(calendarId=calendar_id, body=event).execute()
                logging.debug(f"Event created successfully: {created_event}")
                results.append(f"Event created: {created_event.get('htmlLink')}")
            except HttpError as error:
                logging.error(f"Error creating event: {error}")
                results.append(f"An error occurred: {error}")
        return results

    def list_calendars(self):
        logging.debug("Attempting to list calendars")
        try:
            calendars_result = self.service.calendarList().list().execute()
            calendars = calendars_result.get('items', [])

            if not calendars:
                print('No calendars found.')
                return

            print("Available calendars:")
            for calendar in calendars:
                summary = calendar['summary']
                calendar_id = calendar['id']
                primary = "Primary" if calendar.get('primary') else ""
                print(f"{summary} ({calendar_id}) {primary}")
        except HttpError as error:
            print(f'An error occurred while listing calendars: {error}')

def main():
    print("Welcome to the Continuous Chat Calendar Assistant!")
    print("You can chat normally, and if you mention scheduling, I'll help you create events.")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")

    assistant = ContinuousChatCalendarAssistant()

    assistant.list_calendars()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Assistant: Goodbye!")
            break
        
        try:
            llm_response, scheduling_results = assistant.chat(user_input)
            
            if scheduling_results:
                print("Scheduling results:")
                for result in scheduling_results:
                    print(result)
            print()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            print("The assistant encountered an error. Please check the logs and try again.")

if __name__ == "__main__":
    main()
