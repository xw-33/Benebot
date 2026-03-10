import re
from typing import Optional, Tuple, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

class MultimodalHandler:
    def __init__(self, google_maps_api_key: Optional[str] = None):
        self.google_maps_api_key = google_maps_api_key

        # Regex patterns for fast location detection/extraction
        self.location_patterns = [
            r'where (?:is|are|can I find|do I get|does one find) (.+?)(?:\?|$)',
            r'where\'?s (.+?)(?:\?|$)',
            r'(?:show|find|locate) (?:me )?(?:the )?(?:location|map|directions) (?:of|to) (.+)',
            r'how (?:do I|to) get to (.+?)(?:\?|$)',
            r'(.+) (?:location|address|map|directions)',
            r'location of (.+)',
            r'map of (.+)',
            r'directions to (.+)'
        ]

        # Tiny LLM for fallback tasks
        self.tiny_llm = OllamaLLM(model="phi3:mini")  # fast model

        # Classification prompt (returns 'location' or 'text')
        self.classify_prompt = PromptTemplate.from_template(
            "Classify the query as 'location' or 'text'. Answer with one word.\nQuery: {query}\nType:"
        )
        self.classify_chain = self.classify_prompt | self.tiny_llm

        # Extraction prompt (returns the location name)
        self.extract_prompt = PromptTemplate.from_template(
            "Extract the location name from this query. Reply with ONLY the name.\nQuery: {query}\nLocation:"
        )
        self.extract_chain = self.extract_prompt | self.tiny_llm

    # ---------- Public methods used by chatbot ----------
    def detect_query_type(self, query: str) -> str:
        """Fast regex detection – override if needed."""
        query_lower = query.lower()
        for pattern in self.location_patterns:
            if re.search(pattern, query_lower):
                return "location"
        return "text"

    def extract_location_entity(self, query: str) -> Optional[str]:
        """Fast regex extraction."""
        query_lower = query.lower()
        for pattern in self.location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1).strip()
        return None

    def classify_with_tiny_llm(self, query: str) -> str:
        """Fallback classification using tiny LLM."""
        try:
            response = self.classify_chain.invoke({"query": query}).strip().lower()
            return "location" if "location" in response else "text"
        except Exception:
            return "text"

    def extract_with_tiny_llm(self, query: str) -> Optional[str]:
        """Fallback extraction using tiny LLM."""
        try:
            response = self.extract_chain.invoke({"query": query}).strip()
            return response.strip('"\'') if response else None
        except Exception:
            return None

    # ---------- Map methods (unchanged) ----------
    def get_map_embed(self, location_name: str) -> Tuple[str, Optional[Tuple[float, float]]]:
        if not self.google_maps_api_key:
            search_url = f"https://www.google.com/maps/search/?api=1&query={location_name.replace(' ', '+')}"
            return f'<a href="{search_url}" target="_blank">View {location_name} on Google Maps</a>', None
        try:
            import googlemaps
            gmaps = googlemaps.Client(key=self.google_maps_api_key)
            geocode_result = gmaps.geocode(location_name)
            if geocode_result:
                loc = geocode_result[0]['geometry']['location']
                lat, lng = loc['lat'], loc['lng']
                embed_html = f"""
                    <iframe width="100%" height="400" style="border:0" loading="lazy"
                        allowfullscreen referrerpolicy="no-referrer-when-downgrade"
                        src="https://www.google.com/maps/embed/v1/place?key={self.google_maps_api_key}&q={lat},{lng}&zoom=15">
                    </iframe>
                """
                return embed_html, (lat, lng)
        except Exception:
            pass
        search_url = f"https://www.google.com/maps/search/?api=1&query={location_name.replace(' ', '+')}"
        return f'<a href="{search_url}" target="_blank">View {location_name} on Google Maps</a>', None

    def format_multimodal_response(self, text_response: str, query_type: str,
                                   entity: Optional[str] = None) -> Dict[str, Any]:
        response = {"text": text_response, "type": query_type, "has_multimedia": query_type != "text"}
        if query_type == "location" and entity:
            map_html, coords = self.get_map_embed(entity)
            response["map_html"] = map_html
            response["coordinates"] = coords
            response["location_name"] = entity
        return response