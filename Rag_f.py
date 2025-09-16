import json
from typing import Dict, Any, List
from datetime import datetime

def convert_to_rag_friendly(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert groundwater data into RAG-friendly format with structured information
    """
    
    # Extract key information
    location = data.get("locationName", "Unknown")
    
    # Main summary section
    rag_data = {
        "location": location,
        "category": data.get("category", {}),
        "stage_of_extraction": data.get("stageOfExtraction", {}),
        "water_availability": {
            "total_availability": data.get("totalGWAvailability", {}).get("total", 0),
            "current_availability": data.get("currentAvailabilityForAllPurposes", {}).get("total", 0),
            "future_availability": data.get("availabilityForFutureUse", {}).get("total", 0),
            "command_area_availability": data.get("totalGWAvailability", {}).get("command", 0),
            "non_command_area_availability": data.get("totalGWAvailability", {}).get("non_command", 0)
        },
        "recharge_summary": {
            "total_recharge": data.get("rechargeData", {}).get("total", {}).get("total", 0),
            "rainfall_recharge": data.get("rechargeData", {}).get("rainfall", {}).get("total", 0),
            "agriculture_recharge": data.get("rechargeData", {}).get("agriculture", {}).get("total", 0),
            "surface_irrigation_recharge": data.get("rechargeData", {}).get("surface_irrigation", {}).get("total", 0),
            "gw_irrigation_recharge": data.get("rechargeData", {}).get("gw_irrigation", {}).get("total", 0),
            "canal_recharge": data.get("rechargeData", {}).get("canal", {}).get("total", 0)
        },
        "draft_summary": {
            "total_draft": data.get("draftData", {}).get("total", {}).get("total", 0),
            "agriculture_draft": data.get("draftData", {}).get("agriculture", {}).get("total", 0),
            "domestic_draft": data.get("draftData", {}).get("domestic", {}).get("total", 0),
            "industry_draft": data.get("draftData", {}).get("industry", {}).get("total", 0),
            "draft_intensity": calculate_draft_intensity(data)
        },
        "area_breakdown": {
            "total_area": data.get("area", {}).get("total", {}).get("totalArea", 0),
            "recharge_worthy_area": data.get("area", {}).get("recharge_worthy", {}).get("totalArea", 0),
            "non_recharge_worthy_area": data.get("area", {}).get("non_recharge_worthy", {}).get("totalArea", 0),
            "non_command_area": data.get("area", {}).get("total", {}).get("nonCommandArea", 0)
        },
        "rainfall": {
            "total_rainfall": data.get("rainfall", {}).get("total", 0),
            "rainfall_intensity": calculate_rainfall_intensity(data)
        },
        "losses": {
            "total_loss": data.get("loss", {}).get("total", 0),
            "command_area_loss": data.get("loss", {}).get("command", 0),
            "non_command_area_loss": data.get("loss", {}).get("non_command", 0)
        },
        "status_assessment": {
            "overall_status": data.get("category", {}).get("total", "unknown"),
            "firka_breakdown": data.get("reportSummary", {}).get("total", {}).get("FIRKA", {}),
            "critical_zones": identify_critical_zones(data)
        },
        "allocation": {
            "domestic_allocation": data.get("gwallocation", {}).get("domestic", {}).get("total", 0),
            "industry_allocation": data.get("gwallocation", {}).get("industry", {}).get("total", 0),
            "total_allocation": data.get("gwallocation", {}).get("total", {}).get("total", 0)
        },
        "key_metrics": {
            "extraction_rate_percentage": data.get("stageOfExtraction", {}).get("total", 0),
            "recharge_efficiency": calculate_recharge_efficiency(data),
            "water_balance": calculate_water_balance(data),
            "sustainability_index": calculate_sustainability_index(data),
            "stress_level": assess_stress_level(data)
        },
        "alerts_and_warnings": generate_alerts(data)
    }
    
    # Add additional context for RAG
    rag_data["context"] = generate_context_summary(rag_data)
    rag_data["search_terms"] = generate_search_terms(location, rag_data)
    rag_data["recommendations"] = generate_recommendations(rag_data)
    
    return rag_data

def calculate_recharge_efficiency(data: Dict[str, Any]) -> float:
    """Calculate recharge efficiency percentage"""
    total_recharge = data.get("rechargeData", {}).get("total", {}).get("total", 0)
    rainfall = data.get("rainfall", {}).get("total", 0)
    
    if rainfall > 0:
        return round((total_recharge / rainfall) * 100, 2)
    return 0.0

def calculate_draft_intensity(data: Dict[str, Any]) -> float:
    """Calculate draft intensity per unit area"""
    total_draft = data.get("draftData", {}).get("total", {}).get("total", 0)
    total_area = data.get("area", {}).get("total", {}).get("totalArea", 0)
    
    if total_area > 0:
        return round(total_draft / total_area, 4)
    return 0.0

def calculate_rainfall_intensity(data: Dict[str, Any]) -> float:
    """Calculate rainfall intensity per unit area"""
    rainfall = data.get("rainfall", {}).get("total", 0)
    total_area = data.get("area", {}).get("total", {}).get("totalArea", 0)
    
    if total_area > 0:
        return round(rainfall / total_area, 4)
    return 0.0

def calculate_water_balance(data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate water balance metrics"""
    recharge = data.get("rechargeData", {}).get("total", {}).get("total", 0)
    draft = data.get("draftData", {}).get("total", {}).get("total", 0)
    availability = data.get("totalGWAvailability", {}).get("total", 0)
    loss = data.get("loss", {}).get("total", 0)
    
    return {
        "net_balance": round(recharge - draft - loss, 2),
        "utilization_rate": round((draft / availability) * 100, 2) if availability > 0 else 0,
        "safety_margin": round((availability - draft) / availability * 100, 2) if availability > 0 else 0,
        "deficit": round(draft - recharge, 2) if draft > recharge else 0
    }

def calculate_sustainability_index(data: Dict[str, Any]) -> float:
    """Calculate groundwater sustainability index"""
    recharge = data.get("rechargeData", {}).get("total", {}).get("total", 0)
    draft = data.get("draftData", {}).get("total", {}).get("total", 0)
    
    if recharge > 0:
        return round((recharge / draft) * 100, 2) if draft > 0 else 100
    return 0.0

def assess_stress_level(data: Dict[str, Any]) -> str:
    """Assess water stress level"""
    extraction_rate = data.get("stageOfExtraction", {}).get("total", 0)
    
    if extraction_rate > 100:
        return "critical_stress"
    elif extraction_rate > 85:
        return "high_stress"
    elif extraction_rate > 70:
        return "moderate_stress"
    else:
        return "low_stress"

def identify_critical_zones(data: Dict[str, Any]) -> Dict[str, int]:
    """Identify critical zones from report summary"""
    firka = data.get("reportSummary", {}).get("total", {}).get("FIRKA", {})
    
    return {
        "over_exploited_firkas": firka.get("over_exploited", 0),
        "semi_critical_firkas": firka.get("semi_critical", 0),
        "safe_firkas": firka.get("safe", 0)
    }

def generate_alerts(data: Dict[str, Any]) -> List[str]:
    """Generate alerts based on water status"""
    alerts = []
    category = data.get("category", {}).get("total", "")
    extraction_rate = data.get("stageOfExtraction", {}).get("total", 0)
    
    if category == "over_exploited":
        alerts.append("CRITICAL: Area classified as over-exploited")
    
    if extraction_rate > 100:
        alerts.append(f"ALERT: Extraction rate {extraction_rate}% exceeds recharge")
    
    water_balance = calculate_water_balance(data)
    if water_balance["deficit"] > 0:
        alerts.append(f"WARNING: Water deficit of {water_balance['deficit']} ha-m")
    
    return alerts

def generate_context_summary(rag_data: Dict[str, Any]) -> str:
    """Generate natural language context summary"""
    location = rag_data["location"]
    status = rag_data["status_assessment"]["overall_status"]
    extraction_rate = rag_data["key_metrics"]["extraction_rate_percentage"]
    critical_zones = rag_data["status_assessment"]["critical_zones"]
    
    return (
        f"{location} groundwater assessment shows {status.upper()} status with {extraction_rate}% extraction rate. "
        f"Total available groundwater: {rag_data['water_availability']['total_availability']:,.0f} ha-m. "
        f"Water draft exceeds recharge by {rag_data['key_metrics']['water_balance']['deficit']:,.0f} ha-m. "
        f"Critical situation with {critical_zones['over_exploited_firkas']} over-exploited firkas. "
        f"Immediate conservation measures required."
    )

def generate_search_terms(location: str, rag_data: Dict[str, Any]) -> List[str]:
    """Generate relevant search terms for RAG systems"""
    terms = [
        location.lower(),
        "groundwater assessment",
        "water resources",
        f"{rag_data['status_assessment']['overall_status']} category",
        "aquifer management",
        f"{int(rag_data['key_metrics']['extraction_rate_percentage'])}% extraction"
    ]
    
    # Add status-specific terms
    status = rag_data["status_assessment"]["overall_status"]
    if status == "over_exploited":
        terms.extend(["water emergency", "depletion crisis", "urgent intervention needed"])
    
    return terms

def generate_recommendations(rag_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on water status"""
    recommendations = []
    status = rag_data["status_assessment"]["overall_status"]
    
    if status == "over_exploited":
        recommendations.extend([
            "Implement water rationing measures",
            "Promote drip irrigation and water-efficient agriculture",
            "Enforce regulations on groundwater extraction",
            "Develop artificial recharge structures",
            "Implement water recycling and reuse programs",
            "Conduct awareness campaigns for water conservation"
        ])
    elif status == "critical":
        recommendations.extend([
            "Monitor groundwater levels closely",
            "Implement controlled extraction policies",
            "Promote water-saving technologies",
            "Develop watershed management plans"
        ])
    else:
        recommendations.extend([
            "Continue monitoring groundwater levels",
            "Promote sustainable water use practices",
            "Plan for future water needs"
        ])
    
    return recommendations

def process_groundwater_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to process groundwater data into RAG-friendly format
    """
    try:
        rag_friendly_data = convert_to_rag_friendly(input_data)
        
        # Add metadata
        rag_friendly_data["metadata"] = {
            "processing_timestamp": datetime.now().isoformat(),
            "data_source": "groundwater_assessment",
            "format_version": "1.1",
            "location_uuid": input_data.get("locationUUID", ""),
            "risk_level": "high" if rag_friendly_data["status_assessment"]["overall_status"] in ["over_exploited", "critical"] else "moderate"
        }
        
        return rag_friendly_data
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "original_data": input_data}

# Process all locations in the provided data
if __name__ == "__main__":
    # First, you need to load your groundwater data
    # Replace this with your actual data loading method
    try:
        with open("F:\\Agentic_INGRES\\per.json", "r") as f:
            groundwater_data = json.load(f)
    except FileNotFoundError:
        print("Error: groundwater_data.json file not found")
        print("Please make sure the file exists or provide the data directly")
        exit(1)
    
    # Process each location
    rag_results = {}
    for location_data in groundwater_data:
        if isinstance(location_data, dict) and "locationName" in location_data:
            location_name = location_data["locationName"]
            rag_results[location_name] = process_groundwater_data(location_data)
    
    # Save all results to a file
    with open("all_locations_groundwater_rag.json", "w") as f:
        json.dump(rag_results, f, indent=2)
    
    print("RAG-friendly data processing complete!")
    print(f"Processed {len(rag_results)} locations")
    
    # Print summary for each location
    for location_name, data in rag_results.items():
        if "error" not in data:
            print(f"\n{location_name}:")
            print(f"  Status: {data['status_assessment']['overall_status']}")
            print(f"  Extraction Rate: {data['key_metrics']['extraction_rate_percentage']}%")
            print(f"  Water Deficit: {data['key_metrics']['water_balance']['deficit']:,.0f} ha-m")