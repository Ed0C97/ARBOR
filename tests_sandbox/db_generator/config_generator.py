"""Config Generator - Generates ARBOR configuration files.

Creates both SOURCE_SCHEMA_CONFIG and domain_profile.json for each random schema.
This ensures ARBOR can properly interpret and test against the generated database.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema_generator import DatabaseSchema, TableSchema


# =============================================================================
# VIBE DIMENSIONS PER VERTICAL
# =============================================================================

VIBE_DIMENSIONS = {
    "restaurant": [
        {"name": "atmosfera", "description": "Qualità dell'ambiente e dell'arredamento", "weight": 5},
        {"name": "qualita_cibo", "description": "Gusto, freschezza e presentazione dei piatti", "weight": 5},
        {"name": "servizio", "description": "Cortesia e professionalità del personale", "weight": 4},
        {"name": "rapporto_qualita_prezzo", "description": "Valore percepito rispetto al costo", "weight": 4},
        {"name": "location", "description": "Posizione e accessibilità del locale", "weight": 3},
        {"name": "menu_varieta", "description": "Ampiezza e originalità delle proposte", "weight": 3},
        {"name": "pulizia", "description": "Igiene e cura degli spazi", "weight": 4},
        {"name": "rumorosita", "description": "Livello di rumore e tranquillità", "weight": 2},
    ],
    "hotel": [
        {"name": "comfort_camera", "description": "Qualità del letto, pulizia e dotazioni", "weight": 5},
        {"name": "posizione", "description": "Vicinanza a attrazioni e trasporti", "weight": 4},
        {"name": "servizio_reception", "description": "Efficienza e cortesia dello staff", "weight": 4},
        {"name": "colazione", "description": "Qualità e varietà della colazione", "weight": 3},
        {"name": "rapporto_qualita_prezzo", "description": "Valore percepito rispetto al costo", "weight": 4},
        {"name": "pulizia", "description": "Igiene di camere e aree comuni", "weight": 5},
        {"name": "wifi_connettivita", "description": "Qualità della connessione internet", "weight": 3},
        {"name": "silenziosita", "description": "Tranquillità e isolamento acustico", "weight": 3},
    ],
    "retail": [
        {"name": "qualita_prodotti", "description": "Standard e affidabilità dei prodotti", "weight": 5},
        {"name": "prezzi", "description": "Competitività e trasparenza dei prezzi", "weight": 4},
        {"name": "assistenza_clienti", "description": "Supporto pre e post vendita", "weight": 4},
        {"name": "spedizione", "description": "Velocità e affidabilità delle consegne", "weight": 4},
        {"name": "catalogo", "description": "Ampiezza e varietà dell'offerta", "weight": 3},
        {"name": "facilita_acquisto", "description": "Semplicità del processo d'acquisto", "weight": 3},
        {"name": "resi", "description": "Politica di reso e rimborsi", "weight": 3},
        {"name": "sostenibilita", "description": "Impegno ambientale e sociale", "weight": 2},
    ],
    "spa": [
        {"name": "qualita_trattamenti", "description": "Efficacia e professionalità dei servizi", "weight": 5},
        {"name": "ambiente", "description": "Atmosfera rilassante e curata", "weight": 5},
        {"name": "personale", "description": "Competenza e cortesia degli operatori", "weight": 4},
        {"name": "pulizia", "description": "Igiene di spazi e attrezzature", "weight": 5},
        {"name": "rapporto_qualita_prezzo", "description": "Valore percepito rispetto al costo", "weight": 4},
        {"name": "prodotti", "description": "Qualità dei prodotti utilizzati", "weight": 3},
        {"name": "prenotazione", "description": "Facilità di prenotazione", "weight": 2},
        {"name": "privacy", "description": "Riservatezza e discrezione", "weight": 3},
    ],
    "healthcare": [
        {"name": "competenza_medica", "description": "Preparazione e aggiornamento dei medici", "weight": 5},
        {"name": "tempi_attesa", "description": "Rapidità nelle visite e referti", "weight": 4},
        {"name": "cortesia_personale", "description": "Umanità e disponibilità dello staff", "weight": 4},
        {"name": "pulizia", "description": "Igiene delle strutture", "weight": 5},
        {"name": "tecnologia", "description": "Modernità delle attrezzature", "weight": 3},
        {"name": "accessibilita", "description": "Facilità di accesso e parcheggio", "weight": 3},
        {"name": "comunicazione", "description": "Chiarezza nelle spiegazioni", "weight": 4},
        {"name": "privacy", "description": "Riservatezza dei dati sanitari", "weight": 4},
    ],
    "fitness": [
        {"name": "attrezzature", "description": "Qualità e varietà delle macchine", "weight": 5},
        {"name": "pulizia", "description": "Igiene degli spazi e attrezzi", "weight": 5},
        {"name": "istruttori", "description": "Competenza e disponibilità trainer", "weight": 4},
        {"name": "orari", "description": "Flessibilità degli orari di apertura", "weight": 3},
        {"name": "corsi", "description": "Varietà e qualità delle lezioni", "weight": 3},
        {"name": "spogliatoi", "description": "Comfort e pulizia spogliatoi", "weight": 4},
        {"name": "rapporto_qualita_prezzo", "description": "Valore dell'abbonamento", "weight": 4},
        {"name": "affollamento", "description": "Gestione degli spazi nelle ore di punta", "weight": 3},
    ],
}

QUALITY_ASPECTS = {
    "restaurant": ["gusto", "presentazione", "freschezza", "porzioni", "originalità"],
    "hotel": ["comfort", "silenzio", "vista", "dotazioni", "pulizia"],
    "retail": ["durabilità", "design", "packaging", "funzionalità", "materiali"],
    "spa": ["relax", "efficacia", "durata", "profumi", "benessere"],
    "healthcare": ["accuratezza", "rapidità", "chiarezza", "empatia", "follow-up"],
    "fitness": ["efficacia", "sicurezza", "motivazione", "progressi", "varietà"],
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EntityTypeConfig:
    """Configuration for mapping a table to an ARBOR entity type."""
    entity_type: str
    table_name: str
    id_column: str = "id"
    required_mappings: dict[str, str] = field(default_factory=dict)
    optional_mappings: dict[str, str] = field(default_factory=dict)
    text_fields_for_embedding: list[str] = field(default_factory=list)
    active_filter_column: str | None = None
    active_filter_value: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "table_name": self.table_name,
            "id_column": self.id_column,
            "required_mappings": self.required_mappings,
            "optional_mappings": self.optional_mappings,
            "text_fields_for_embedding": self.text_fields_for_embedding,
            "active_filter_column": self.active_filter_column,
            "active_filter_value": self.active_filter_value,
        }


@dataclass
class DomainProfile:
    """Complete domain profile for ARBOR."""
    vertical: str
    language: str = "it"
    entity_types: list[str] = field(default_factory=list)
    vibe_dimensions: list[dict] = field(default_factory=list)
    quality_aspects: list[str] = field(default_factory=list)
    calibration_samples: dict[str, list] = field(default_factory=dict)
    persona: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "vertical": self.vertical,
            "language": self.language,
            "entity_types": self.entity_types,
            "vibe_dimensions": self.vibe_dimensions,
            "quality_aspects": self.quality_aspects,
            "calibration_samples": self.calibration_samples,
            "persona": self.persona,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class GeneratedConfigs:
    """Container for all generated configs for a schema."""
    schema_name: str
    vertical: str
    
    # SOURCE_SCHEMA_CONFIG content
    schema_config: list[EntityTypeConfig] = field(default_factory=list)
    
    # domain_profile.json content
    domain_profile: DomainProfile | None = None
    
    def get_schema_config_json(self) -> str:
        """Get SOURCE_SCHEMA_CONFIG as JSON string."""
        return json.dumps(
            [c.to_dict() for c in self.schema_config],
            indent=2,
            ensure_ascii=False,
        )
    
    def get_domain_profile_json(self) -> str:
        """Get domain_profile as JSON string."""
        if self.domain_profile:
            return self.domain_profile.to_json()
        return "{}"


# =============================================================================
# CONFIG GENERATOR
# =============================================================================

class ConfigGenerator:
    """Generates ARBOR configuration from database schema."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
    
    def generate(
        self,
        schema: DatabaseSchema,
        sample_data: dict[str, list[dict]] | None = None,
    ) -> GeneratedConfigs:
        """Generate complete ARBOR configuration for a schema.
        
        Args:
            schema: Database schema to generate config for
            sample_data: Optional sample rows per table for calibration
        
        Returns:
            GeneratedConfigs with schema_config and domain_profile
        """
        configs = GeneratedConfigs(
            schema_name=schema.schema_name,
            vertical=schema.vertical,
        )
        
        # 1. Generate SOURCE_SCHEMA_CONFIG (entity type mappings)
        configs.schema_config = self._generate_schema_config(schema)
        
        # 2. Generate domain_profile
        configs.domain_profile = self._generate_domain_profile(
            schema, sample_data
        )
        
        return configs
    
    def _generate_schema_config(self, schema: DatabaseSchema) -> list[EntityTypeConfig]:
        """Generate SOURCE_SCHEMA_CONFIG for mapping tables to entity types."""
        configs = []
        
        for table in schema.tables:
            # Determine which columns map to what
            columns = {c.name: c for c in table.columns}
            
            # Required mappings
            required = {}
            if "name" in columns:
                required["name"] = "name"
            elif any("name" in c.lower() for c in columns):
                name_col = next(c for c in columns if "name" in c.lower())
                required["name"] = name_col
            else:
                # Use first VARCHAR column as name
                for col in table.columns:
                    if "VARCHAR" in col.data_type.upper():
                        required["name"] = col.name
                        break
            
            # Optional mappings
            optional = {}
            text_fields = []
            
            for col_name, col in columns.items():
                col_lower = col_name.lower()
                
                # Map common fields
                if "description" in col_lower:
                    optional["description"] = col_name
                    text_fields.append(col_name)
                elif "category" in col_lower or "type" in col_lower:
                    optional["category"] = col_name
                elif "city" in col_lower:
                    optional["city"] = col_name
                elif "address" in col_lower:
                    optional["address"] = col_name
                elif "price" in col_lower:
                    optional["price"] = col_name
                elif "rating" in col_lower:
                    optional["rating"] = col_name
                elif "status" in col_lower:
                    optional["status"] = col_name
                
                # Text fields for embedding
                if col.data_type.upper() in ("TEXT", "VARCHAR(255)"):
                    if col_name not in text_fields and "id" not in col_lower:
                        text_fields.append(col_name)
            
            # Ensure name is in text fields
            if required.get("name") and required["name"] not in text_fields:
                text_fields.insert(0, required["name"])
            
            # Active filter
            active_col = None
            for col_name in columns:
                if col_name.lower() in ("is_active", "active", "status", "published"):
                    active_col = col_name
                    break
            
            configs.append(EntityTypeConfig(
                entity_type=table.name,
                table_name=table.name,
                id_column="id",
                required_mappings=required,
                optional_mappings=optional,
                text_fields_for_embedding=text_fields[:5],  # Limit to 5
                active_filter_column=active_col,
                active_filter_value=True,
            ))
        
        return configs
    
    def _generate_domain_profile(
        self,
        schema: DatabaseSchema,
        sample_data: dict[str, list[dict]] | None = None,
    ) -> DomainProfile:
        """Generate domain_profile.json for the schema."""
        vertical = schema.vertical
        
        # Get vibe dimensions for this vertical
        dimensions = VIBE_DIMENSIONS.get(vertical, VIBE_DIMENSIONS["restaurant"])
        
        # Select a subset (6-8 dimensions)
        num_dims = self.rng.randint(6, min(8, len(dimensions)))
        selected_dims = self.rng.sample(dimensions, num_dims)
        
        # Get quality aspects
        aspects = QUALITY_ASPECTS.get(vertical, QUALITY_ASPECTS["restaurant"])
        
        # Entity types from tables
        entity_types = [t.name for t in schema.tables]
        
        # Generate calibration samples from sample_data
        calibration = {"best": [], "average": []}
        
        if sample_data:
            for table_name, rows in sample_data.items():
                if rows:
                    # Pick best examples (first 2-3)
                    for row in rows[:3]:
                        if "name" in row:
                            calibration["best"].append({
                                "entity_type": table_name,
                                "name": row.get("name", ""),
                                "description": row.get("description", "")[:200] if row.get("description") else "",
                            })
                    
                    # Pick average examples (middle)
                    mid = len(rows) // 2
                    for row in rows[mid:mid+2]:
                        if "name" in row:
                            calibration["average"].append({
                                "entity_type": table_name,
                                "name": row.get("name", ""),
                            })
        else:
            # Generate fake calibration samples
            from faker import Faker
            fake = Faker("it_IT")
            fake.seed_instance(self.seed)
            
            for table in schema.tables[:2]:
                calibration["best"].append({
                    "entity_type": table.name,
                    "name": f"Esempio Eccellente - {fake.company()}",
                    "description": fake.paragraph(nb_sentences=2),
                })
                calibration["average"].append({
                    "entity_type": table.name,
                    "name": f"Esempio Medio - {fake.company()}",
                })
        
        # Persona for the vertical
        persona = self._generate_persona(vertical)
        
        return DomainProfile(
            vertical=vertical,
            language="it",
            entity_types=entity_types,
            vibe_dimensions=selected_dims,
            quality_aspects=aspects,
            calibration_samples=calibration,
            persona=persona,
        )
    
    def _generate_persona(self, vertical: str) -> dict[str, Any]:
        """Generate persona config for the vertical."""
        personas = {
            "restaurant": {
                "name": "Gusto",
                "role": "esperto gastronomico",
                "tone": "appassionato e competente",
                "focus": "qualità del cibo e dell'esperienza culinaria",
            },
            "hotel": {
                "name": "Hospitality",
                "role": "concierge esperto",
                "tone": "elegante e attento ai dettagli",
                "focus": "comfort e qualità del soggiorno",
            },
            "retail": {
                "name": "Shopper",
                "role": "consulente acquisti",
                "tone": "pratico e obiettivo",
                "focus": "qualità e valore dei prodotti",
            },
            "spa": {
                "name": "Wellness",
                "role": "esperto di benessere",
                "tone": "rilassante e professionale",
                "focus": "qualità dei trattamenti e relax",
            },
            "healthcare": {
                "name": "Salus",
                "role": "consulente sanitario",
                "tone": "rassicurante e competente",
                "focus": "qualità delle cure e attenzione al paziente",
            },
            "fitness": {
                "name": "Trainer",
                "role": "coach sportivo",
                "tone": "motivante e energico",
                "focus": "risultati e benessere fisico",
            },
        }
        return personas.get(vertical, personas["restaurant"])


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    from .schema_generator import SchemaGenerator
    
    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate(vertical="restaurant", num_tables=3)
    
    # Generate configs
    config_gen = ConfigGenerator(seed=42)
    configs = config_gen.generate(schema)
    
    print("=" * 60)
    print("SOURCE_SCHEMA_CONFIG:")
    print("=" * 60)
    print(configs.get_schema_config_json())
    
    print("\n" + "=" * 60)
    print("domain_profile.json:")
    print("=" * 60)
    print(configs.get_domain_profile_json())
