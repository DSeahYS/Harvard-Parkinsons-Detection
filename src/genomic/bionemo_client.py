import numpy as np
from datetime import datetime

class BioNeMoSimulator:
    """Simulates genomic analysis for Parkinson's disease risk assessment"""
    
    # Singapore-specific genetic risk factors (From SG-GWAS studies)
    ETHNIC_ADJUSTMENTS = {
        'chinese': {'LRRK2': 1.05, 'GBA': 1.12},
        'malay': {'LRRK2': 1.18, 'GBA': 1.25},
        'indian': {'LRRK2': 1.15, 'GBA': 1.20},
        'default': {'LRRK2': 1.0, 'GBA': 1.0}
    }
    
    # PD-associated variants and base risk contributions
    VARIANTS = {
        'LRRK2': {
            'variant': 'G2019S',
            'base_risk': 2.4,
            'prevalence': {'chinese': 0.012, 'malay': 0.008, 'indian': 0.010}
        },
        'GBA': {
            'variant': 'N370S', 
            'base_risk': 5.4,
            'prevalence': {'chinese': 0.005, 'malay': 0.003, 'indian': 0.004}
        },
        'SNCA': {
            'variant': 'A53T',
            'base_risk': 8.1,
            'prevalence': {'chinese': 0.001, 'malay': 0.0005, 'indian': 0.0008}
        }
    }

    def analyze_genomics(self, risk_level, ethnicity='chinese'):
        """Simulate genomic analysis based on eye tracking risk and ethnicity"""
        variants = {}
        
        # Generate variants based on risk level
        if risk_level > 0.3:
            variants.update(self._generate_variant('LRRK2', risk_level, ethnicity))
        if risk_level > 0.5:
            variants.update(self._generate_variant('GBA', risk_level, ethnicity))
        if risk_level > 0.7:
            variants.update(self._generate_variant('SNCA', risk_level, ethnicity))
            
        return {
            'patient_variants': variants,
            'genomic_risk_score': self._calculate_genomic_risk(variants),
            'model_used': 'BioNeMo Evo-2 (Simulated SG Edition)',
            'analysis_date': datetime.now().isoformat()
        }
    
    def _generate_variant(self, gene, risk_level, ethnicity):
        """Generate simulated variant data"""
        base = self.VARIANTS[gene]
        # Ensure ethnicity exists, fallback to default if not
        ethnic_adj_map = self.ETHNIC_ADJUSTMENTS.get(ethnicity, self.ETHNIC_ADJUSTMENTS['default'])
        adj = ethnic_adj_map.get(gene, 1.0)
        
        # Ensure prevalence exists for the ethnicity, fallback if needed (though ideally data should be complete)
        prevalence = base['prevalence'].get(ethnicity, base['prevalence'].get('chinese', 0)) 

        return {
            gene: {
                'variant': base['variant'],
                'risk_contribution': base['base_risk'] * risk_level * adj,
                'population_prevalence': prevalence,
                'sg_reference': f"SG-PD-GENOME v2.1 ({ethnicity} cohort)"
            }
        }
    
    def _calculate_genomic_risk(self, variants):
        """Calculate composite genomic risk score"""
        if not variants:
            return 0.0
        
        # Weighted sum of risk contributions
        total_risk = sum(v['risk_contribution'] for v in variants.values())
        return min(total_risk / 7.5, 1.0)  # Normalized to 0-1 scale

# Example usage
if __name__ == "__main__":
    simulator = BioNeMoSimulator()
    
    # Test case for Chinese patient with medium risk
    print("Chinese Patient (Risk 0.6):")
    print(simulator.analyze_genomics(0.6, 'chinese'))
    
    # Test case for Malay patient with high risk
    print("\nMalay Patient (Risk 0.8):")
    print(simulator.analyze_genomics(0.8, 'malay'))

    # Test case for unknown ethnicity (should use default)
    print("\nUnknown Ethnicity Patient (Risk 0.7):")
    print(simulator.analyze_genomics(0.7, 'other'))
