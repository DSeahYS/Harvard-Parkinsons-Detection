# C:\Users\Dave\PycharmProjects\GenomeGuard\src\genomic\bionemo_client.py
import random
import logging
import os
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BioNeMoClient:
    """
    Simulated BioNeMo client for genomic risk analysis of Parkinson's Disease,
    incorporating NVIDIA BioNeMo framework concepts and real PD genetic research.
    """
    def __init__(self, api_key=None):
        """
        Initializes the BioNeMo client with PD genetic database.

        Args:
            api_key (str, optional): API key for the BioNeMo service.
                                     Defaults to environment variable 'BIONEMO_API_KEY'.
        """
        logging.info("BioNeMo client initialized (Simulating NVIDIA BioNeMo Framework)")
        
        # --- Ethnicity Risk Adjustment Factors ---
        self.ethnicity_adjustments = {
            'chinese': {'LRRK2': 1.12, 'GBA': 1.05, 'global': 1.08},
            'malay':   {'LRRK2': 1.25, 'GBA': 1.18, 'global': 1.20},
            'indian':  {'LRRK2': 1.20, 'GBA': 1.15, 'global': 1.18},
            'default': {'LRRK2': 1.00, 'GBA': 1.00, 'global': 1.00}
        }
        
        # --- PD Gene Database ---
        # Based on search results #3, #4, #5, #7, #8
        self.pd_genes = self._init_pd_gene_database()
        
        # --- Risk Level Thresholds ---
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
        }
        
        # --- Pathway Interactions ---
        self.pathways = {
            'mitochondrial': ['PINK1', 'PRKN', 'PARK7'],
            'lysosomal': ['GBA', 'ATP13A2'],
            'kinase_signaling': ['LRRK2'],
            'trafficking': ['VPS35', 'DNAJC6', 'SYNJ1'],
            'protein_aggregation': ['SNCA', 'FBXO7']
        }

    def _init_pd_gene_database(self):
        """Initialize comprehensive PD gene database based on latest research"""
        return {
            # Autosomal Dominant Genes
            'LRRK2': {
                'inheritance': 'dominant',
                'variants': {
                    'G2019S': {'population_freq': 0.03, 'odds_ratio': 2.5, 'effect': 'pathogenic'},
                    'R1441G': {'population_freq': 0.01, 'odds_ratio': 2.2, 'effect': 'pathogenic'},
                    'E356K': {'population_freq': 0.005, 'odds_ratio': 1.8, 'effect': 'pathogenic'}
                },
                'pathway': 'Kinase signaling',
                'base_risk': 0.03,  # ~3% in PD cohorts from search result #5
                'phenotype': 'typical PD'
            },
            'SNCA': {
                'inheritance': 'dominant',
                'variants': {
                    'A53T': {'population_freq': 0.005, 'odds_ratio': 3.2, 'effect': 'pathogenic'},
                    'E46K': {'population_freq': 0.002, 'odds_ratio': 3.0, 'effect': 'pathogenic'},
                    'H50Q': {'population_freq': 0.001, 'odds_ratio': 2.8, 'effect': 'pathogenic'}
                },
                'pathway': 'Alpha-synuclein aggregation',
                'base_risk': 0.002,
                'phenotype': 'early-onset to LB dementia'
            },
            'VPS35': {
                'inheritance': 'dominant',
                'variants': {
                    'D620N': {'population_freq': 0.006, 'odds_ratio': 2.3, 'effect': 'pathogenic'}
                },
                'pathway': 'Trafficking',
                'base_risk': 0.006,
                'phenotype': 'typical PD'
            },
            
            # Autosomal Recessive Genes - Typical PD
            'PRKN': {
                'inheritance': 'recessive',
                'variants': {
                    'R42P': {'population_freq': 0.02, 'odds_ratio': 1.5, 'effect': 'pathogenic'},
                    'exon_deletions': {'population_freq': 0.01, 'odds_ratio': 2.0, 'effect': 'pathogenic'}
                },
                'pathway': 'Mitochondrial function',
                'base_risk': 0.02,
                'phenotype': 'early-onset typical PD'
            },
            'PINK1': {
                'inheritance': 'recessive',
                'variants': {
                    'Q456X': {'population_freq': 0.015, 'odds_ratio': 1.7, 'effect': 'pathogenic'}
                },
                'pathway': 'Mitochondrial function',
                'base_risk': 0.015,
                'phenotype': 'early-onset typical PD'
            },
            'PARK7': {  # Also known as DJ-1
                'inheritance': 'recessive',
                'variants': {
                    'L166P': {'population_freq': 0.008, 'odds_ratio': 1.9, 'effect': 'pathogenic'}
                },
                'pathway': 'Mitochondrial function',
                'base_risk': 0.008,
                'phenotype': 'early-onset typical PD'
            },
            
            # Atypical Parkinsonism Genes
            'ATP13A2': {
                'inheritance': 'recessive',
                'variants': {
                    'T12M': {'population_freq': 0.004, 'odds_ratio': 2.1, 'effect': 'pathogenic'}
                },
                'pathway': 'Lysosomal function',
                'base_risk': 0.004,
                'phenotype': 'atypical parkinsonism'
            },
            'FBXO7': {
                'inheritance': 'recessive',
                'variants': {
                    'R498X': {'population_freq': 0.003, 'odds_ratio': 2.0, 'effect': 'pathogenic'}
                },
                'pathway': 'Protein degradation',
                'base_risk': 0.003,
                'phenotype': 'atypical parkinsonism'
            },
            'DNAJC6': {
                'inheritance': 'recessive',
                'variants': {
                    'Q734X': {'population_freq': 0.002, 'odds_ratio': 2.5, 'effect': 'pathogenic'},
                    'missense': {'population_freq': 0.004, 'odds_ratio': 1.5, 'effect': 'likely pathogenic'}
                },
                'pathway': 'Trafficking',
                'base_risk': 0.003,
                'phenotype': 'varies by mutation type'
            },
            'SYNJ1': {
                'inheritance': 'recessive',
                'variants': {
                    'R258Q': {'population_freq': 0.001, 'odds_ratio': 2.2, 'effect': 'pathogenic'}
                },
                'pathway': 'Trafficking',
                'base_risk': 0.001,
                'phenotype': 'atypical parkinsonism'
            },
            
            # Risk Factor Genes
            'GBA': {
                'inheritance': 'risk_factor',
                'variants': {
                    'L444P': {'population_freq': 0.03, 'odds_ratio': 1.8, 'effect': 'risk_factor'},
                    'N370S': {'population_freq': 0.05, 'odds_ratio': 1.5, 'effect': 'risk_factor'}
                },
                'pathway': 'Lysosomal function',
                'base_risk': 0.10,  # ~10% in PD cohorts from search result #5
                'phenotype': 'typical PD with cognitive features'
            }
        }

    def _apply_ethnicity_adjustments(self, ethnicity):
        """Apply ethnicity-specific risk adjustments to gene database"""
        ethnicity_lower = ethnicity.lower() if ethnicity else 'default'
        adjustments = self.ethnicity_adjustments.get(ethnicity_lower, self.ethnicity_adjustments['default'])
        
        adjusted_db = self.pd_genes.copy()
        
        # Apply specific gene adjustments where available, or global adjustment otherwise
        for gene in adjusted_db:
            if gene in adjustments:
                gene_factor = adjustments[gene]
            else:
                gene_factor = adjustments['global']
                
            adjusted_db[gene]['ethnicity_factor'] = gene_factor
            
            # Adjust variant frequencies
            for variant in adjusted_db[gene]['variants']:
                variant_data = adjusted_db[gene]['variants'][variant]
                adjusted_freq = min(variant_data['population_freq'] * gene_factor, 1.0)
                adjusted_db[gene]['variants'][variant]['adjusted_freq'] = adjusted_freq
        
        return adjusted_db

    def _simulate_genomic_profile(self, risk_level, ethnicity_adjusted_db):
        """Simulate genomic profile based on eye risk level and ethnicity factors"""
        # Determine risk multiplier based on eye metric risk level
        if risk_level < self.risk_thresholds['low']:
            risk_multiplier = 0.5  # Lower probability of variants
        elif risk_level < self.risk_thresholds['medium']:
            risk_multiplier = 1.0  # Baseline probability
        else:
            risk_multiplier = 2.0  # Higher probability of variants
        
        detected_variants = []
        
        # Determine which variants are present based on adjusted frequencies
        for gene, gene_data in ethnicity_adjusted_db.items():
            for variant_name, variant_data in gene_data['variants'].items():
                # Calculate probability of having this variant
                variant_prob = variant_data['adjusted_freq'] * risk_multiplier
                
                # Random draw to determine if variant is present
                if random.random() < variant_prob:
                    detected_variants.append({
                        'gene': gene,
                        'variant': variant_name,
                        'odds_ratio': variant_data['odds_ratio'],
                        'effect': variant_data['effect'],
                        'pathway': gene_data['pathway']
                    })
        
        return detected_variants

    def _calculate_pathway_burden(self, variants):
        """Calculate pathway burden based on detected variants"""
        pathway_burden = defaultdict(float)
        
        for variant in variants:
            gene = variant['gene']
            
            # Find which pathway this gene belongs to
            for pathway, genes in self.pathways.items():
                if gene in genes:
                    # Add the odds ratio to the pathway burden
                    pathway_burden[pathway] += variant['odds_ratio'] - 1.0  # Subtract 1 to get excess risk
        
        return dict(pathway_burden)

    def _calculate_combined_risk(self, variants, ethnicity):
        """Calculate combined genetic risk using multiplicative model"""
        if not variants:
            return 1.0
            
        # Use multiplicative model for overall risk
        combined_or = 1.0
        for variant in variants:
            combined_or *= variant['odds_ratio']
            
        # Cap at reasonable maximum (10x baseline risk)
        return min(combined_or, 10.0)

    def analyze_genomics(self, eye_risk_level, ethnicity):
        """
        Performs genomic risk analysis based on eye metrics risk level and ethnicity.
        Simulates NVIDIA BioNeMo framework analysis of genetic variants.

        Args:
            eye_risk_level (float): The risk level calculated by PDDetector (0.0 to 1.0).
            ethnicity (str): The patient's ethnicity.

        Returns:
            dict: Results from the genomic simulation.
        """
        logging.info(f"Performing BioNeMo genomic analysis. Eye Risk: {eye_risk_level:.3f}, Ethnicity: {ethnicity}")

        if eye_risk_level is None or ethnicity is None:
             logging.warning("Cannot perform genomic analysis without eye risk level and ethnicity.")
             return {
                 'error': 'Missing eye_risk_level or ethnicity',
                 'combined_risk_score': 1.0,
                 'variants_detected': [],
                 'pathway_analysis': {}
             }

        # Apply ethnicity-specific adjustments to gene database
        ethnicity_adjusted_db = self._apply_ethnicity_adjustments(ethnicity)
        
        # Simulate genomic profile
        detected_variants = self._simulate_genomic_profile(eye_risk_level, ethnicity_adjusted_db)
        
        # Calculate pathway burden
        pathway_burden = self._calculate_pathway_burden(detected_variants)
        
        # Calculate combined risk score
        combined_risk = self._calculate_combined_risk(detected_variants, ethnicity)
        
        # Generate population comparison data
        pop_comparison = {
            'variant_count': len(detected_variants),
            'population_average': 2.1,  # Average number of PD-related variants in general population
            'percentile': min(90, int(len(detected_variants) * 20))  # Simple percentile calculation
        }
        
        # Determine most affected pathways
        dominant_pathways = []
        if pathway_burden:
            max_burden = max(pathway_burden.values())
            dominant_pathways = [p for p, b in pathway_burden.items() if b > 0.5 * max_burden]
        
        results = {
            'combined_risk_score': combined_risk,
            'variants_detected': detected_variants,
            'pathway_analysis': pathway_burden,
            'dominant_pathways': dominant_pathways,
            'population_comparison': pop_comparison,
            'ethnicity_adjustment_applied': self.ethnicity_adjustments.get(
                ethnicity.lower() if ethnicity else 'default', 
                self.ethnicity_adjustments['default']
            )
        }
        
        # Add specific gene risk metrics for backward compatibility
        results['simulated_lrrk2_risk'] = next(
            (v['odds_ratio'] for v in detected_variants if v['gene'] == 'LRRK2'), 
            1.0
        ) - 1.0
        
        results['simulated_gba_risk'] = next(
            (v['odds_ratio'] for v in detected_variants if v['gene'] == 'GBA'), 
            1.0
        ) - 1.0
        
        results['combined_genetic_risk'] = min((results['simulated_lrrk2_risk'] + 
                                              results['simulated_gba_risk'] + 1.0), 
                                             combined_risk)
        
        if detected_variants:
            if any(v['effect'] == 'pathogenic' for v in detected_variants):
                results['variant_profile'] = 'high_risk_profile'
            else:
                results['variant_profile'] = 'moderate_risk_profile'
        else:
            results['variant_profile'] = 'low_risk_profile'
            
        logging.info(f"BioNeMo genomic analysis complete. Detected {len(detected_variants)} variants.")
        
        return results

# Example Usage
if __name__ == '__main__':
    client = BioNeMoClient()

    print("\n--- Testing BioNeMo Client (NVIDIA BioNeMo Framework Simulation) ---")

    risk_levels = [0.1, 0.4, 0.8]
    ethnicities = ['chinese', 'malay', 'indian', 'caucasian', None]

    for risk in risk_levels:
        print(f"\n--- Eye Risk Level: {risk:.2f} ---")
        for eth in ethnicities:
            print(f"  Ethnicity: {eth}")
            results = client.analyze_genomics(risk, eth)
            print(f"    Combined Risk Score: {results['combined_risk_score']:.2f}x")
            print(f"    Variants Detected: {len(results['variants_detected'])}")
            if results['variants_detected']:
                print(f"    Key Variants: {', '.join([f'{v['gene']}-{v['variant']}' for v in results['variants_detected'][:3]])}")
            print(f"    Dominant Pathways: {', '.join(results['dominant_pathways'])}")
            print("-" * 40)
