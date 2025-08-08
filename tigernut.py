Python Codes
import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ProcessParameters:
    daily_capacity: float = 2000
    operating_days: int = 330
    oil_content: float = 0.38
    extraction_efficiency: float = 0.9023
    meal_recovery_efficiency: float = 0.95
    throughput_capacity: float = 9.9
    extraction_loss: float = 1.0

@dataclass
class EquipmentCosts:
    cleaning: float = 69000
    washing: float = 45000
    screw_press: float = 61000
    vacuum_dryer: float = 71000
    storage_filtration: float = 75000
    conveyors: float = 3000
    heater: float = 11000

@dataclass
class OperatingCostParams:
    raw_material_cost: float = 5.00
    electricity_cost: float = 0.10
    annual_electricity: float = 258.84
    steam_cost: float = 32
    annual_steam: float = 55.48
    basic_labor_rate: float = 30.00
    fringe_benefits: float = 0.4
    supervision: float = 0.2
    operational_supplies: float = 0.1
    administration: float = 0.6
    annual_labor_hours: float = 23760
    lab_qc_qa_factor: float = 0.15
    water_cost: float = 0.002 # per kg
    daily_water_consumption: float = 100 # kg/day

@dataclass
class FinancialParams:
    project_lifespan: int = 15
    discount_rate: float = 0.10
    depreciation_method: str = "straight-line"
    tax_rate: float = 0.28
    capacity_utilization_year1: float = 1.00
    capacity_utilization_year3plus: float = 1.00
    oil_price: float = 30.00
    cake_price: float = 1.80
    working_capital_factor: float = 0.15
    startup_capital_factor: float = 0.05

class TigernutOilTEA:
    def __init__(self):
        self.process = ProcessParameters()
        self.equipment = EquipmentCosts()
        self.operating = OperatingCostParams()
        self.financial = FinancialParams()

    def calculate_mass_balance(self) -> Dict[str, float]:
        annual_input = self.process.daily_capacity * self.process.operating_days
        annual_oil = annual_input * self.process.oil_content * self.process.extraction_efficiency
        annual_cake = annual_input * (1 - self.process.oil_content +
                                      self.process.oil_content * (1 - self.process.extraction_efficiency)) * \
                                      self.process.meal_recovery_efficiency
        annual_losses = annual_input - annual_oil - annual_cake

        return {
            'annual_input': annual_input,
            'annual_oil': annual_oil,
            'annual_cake': annual_cake,
            'annual_losses': annual_losses,
            'oil_yield_percentage': (annual_oil / annual_input) * 100,
            'cake_yield_percentage': (annual_cake / annual_input) * 100
        }

    def scale_equipment_cost(self, base_cost: float, new_capacity: float,
                            base_capacity: float, scaling_exponent: float = 0.7) -> float:
        return base_cost * (new_capacity / base_capacity) ** scaling_exponent

    def calculate_capital_costs(self) -> Dict[str, float]:
        equipment_costs = {
            'cleaning': self.equipment.cleaning,
            'washing': self.equipment.washing,
            'screw_press': self.equipment.screw_press,
            'vacuum_dryer': self.equipment.vacuum_dryer,
            'storage_filtration': self.equipment.storage_filtration,
            'conveyors': self.equipment.conveyors,
            'heater': self.equipment.heater
        }

        tepc = sum(equipment_costs.values())
        installation_factor = 1.8
        direct_cost = 2.8 * tepc

        engineering_cost = 0.25 * direct_cost
        construction_cost = 0.35 * direct_cost
        indirect_cost = engineering_cost + construction_cost

        total_plant_cost = direct_cost + indirect_cost

        contractors_fee = 0.05 * total_plant_cost
        contingency = 0.10 * total_plant_cost
        other_cost = contractors_fee + contingency

        fci = total_plant_cost + other_cost
        working_capital = self.financial.working_capital_factor * fci
        startup_capital = self.financial.startup_capital_factor * fci

        tci = fci + working_capital + startup_capital

        return {
            'equipment_costs': equipment_costs,
            'tepc': tepc,
            'direct_cost': direct_cost,
            'indirect_cost': indirect_cost,
            'total_plant_cost': total_plant_cost,
            'other_cost': other_cost,
            'fci': fci,
            'working_capital': working_capital,
            'startup_capital': startup_capital,
            'tci': tci
        }

    def calculate_operating_costs(self, mass_balance: Dict) -> Dict[str, float]:
        raw_material_cost = mass_balance['annual_input'] * self.operating.raw_material_cost

        electricity_cost = self.operating.annual_electricity * self.operating.electricity_cost * 1000
        steam_cost = self.operating.annual_steam * self.operating.steam_cost
        annual_water_consumption = self.operating.daily_water_consumption * self.process.operating_days
        water_cost = annual_water_consumption * self.operating.water_cost
        utility_cost = electricity_cost + steam_cost + water_cost

        labor_rate = self.operating.basic_labor_rate * (1 +
                                                         self.operating.fringe_benefits +
                                                         self.operating.supervision +
                                                         self.operating.operational_supplies +
                                                         self.operating.administration)

        total_labor_cost = self.operating.annual_labor_hours * labor_rate
        lab_qc_qa_cost = self.operating.lab_qc_qa_factor * total_labor_cost

        capital_costs = self.calculate_capital_costs()
        maintenance_cost = 0.05 * capital_costs['fci']

        direct_operating_cost = raw_material_cost + utility_cost + total_labor_cost + lab_qc_qa_cost + maintenance_cost
        overhead = 0.15 * direct_operating_cost

        total_operating_cost = direct_operating_cost + overhead

        return {
            'raw_material_cost': raw_material_cost,
            'utility_cost': utility_cost,
            'total_labor_cost': total_labor_cost,
            'lab_qc_qa_cost': lab_qc_qa_cost,
            'maintenance_cost': maintenance_cost,
            'overhead': overhead,
            'total_operating_cost': total_operating_cost
        }

    def calculate_revenues(self, mass_balance: Dict, year: int) -> Dict[str, float]:
        if year == 1:
            capacity_utilization = self.financial.capacity_utilization_year1
        else:
            capacity_utilization = self.financial.capacity_utilization_year3plus

        oil_revenue = mass_balance['annual_oil'] * self.financial.oil_price * capacity_utilization
        cake_revenue = mass_balance['annual_cake'] * self.financial.cake_price * capacity_utilization

        return {'total_revenue': oil_revenue + cake_revenue,
                'oil_revenue': oil_revenue,
                'cake_revenue': cake_revenue}

    def calculate_cash_flows(self) -> pd.DataFrame:
        mass_balance = self.calculate_mass_balance()
        capital_costs = self.calculate_capital_costs()
        operating_costs = self.calculate_operating_costs(mass_balance)

        years = range(0, self.financial.project_lifespan + 1)
        cash_flows = []

        for year in years:
            if year == 0:
                capex = -capital_costs['tci']
                revenue_details = self.calculate_revenues(mass_balance, year)
                revenue = revenue_details['total_revenue'] # Should be 0 for year 0
                oil_revenue = revenue_details['oil_revenue'] # Should be 0 for year 0
                cake_revenue = revenue_details['cake_revenue'] # Should be 0 for year 0
                opex = 0
                depreciation = 0
                ebitda = 0
                ebit = 0
                tax = 0
                net_income = 0
                cash_flow = capex
            else:
                capex = 0
                revenue_details = self.calculate_revenues(mass_balance, year)
                revenue = revenue_details['total_revenue']
                oil_revenue = revenue_details['oil_revenue']
                cake_revenue = revenue_details['cake_revenue']

                if year == 1:
                    opex = operating_costs['total_operating_cost'] * self.financial.capacity_utilization_year1
                else:
                    opex = operating_costs['total_operating_cost'] * self.financial.capacity_utilization_year3plus

                depreciation = capital_costs['fci'] / self.financial.project_lifespan
                ebitda = revenue - opex
                ebit = ebitda - depreciation
                tax = max(0, ebit * self.financial.tax_rate)
                net_income = ebit - tax
                cash_flow = net_income + depreciation

                if year == self.financial.project_lifespan:
                    cash_flow += capital_costs['working_capital']

            cash_flows.append({
                'Year': year,
                'CAPEX': capex,
                'Total Revenue': revenue,
                'Oil Revenue': oil_revenue,
                'Cake Revenue': cake_revenue,
                'OPEX': opex,
                'Depreciation': depreciation,
                'EBITDA': ebitda,
                'EBIT': ebit,
                'Tax': tax,
                'Net Income': net_income,
                'Cash Flow': cash_flow
            })

        return pd.DataFrame(cash_flows)

    def calculate_npv(self, cash_flows: pd.DataFrame) -> float:
        discount_factors = [(1 + self.financial.discount_rate) ** -year
                           for year in cash_flows['Year']]
        discounted_cash_flows = cash_flows['Cash Flow'] * discount_factors
        return discounted_cash_flows.sum()

    def calculate_irr(self, cash_flows: pd.DataFrame) -> float:
        return npf.irr(cash_flows['Cash Flow'].values)

    def calculate_payback_period(self, cash_flows: pd.DataFrame) -> float:
        cumulative_cash_flow = cash_flows['Cash Flow'].cumsum()
        positive_years = cumulative_cash_flow[cumulative_cash_flow > 0]

        if len(positive_years) == 0:
            return np.inf

        payback_year = positive_years.index[0]

        if payback_year == 0:
            return 0

        previous_cumulative = cumulative_cash_flow.iloc[payback_year - 1]
        current_year_flow = cash_flows['Cash Flow'].iloc[payback_year]

        fractional_year = -previous_cumulative / current_year_flow

        return payback_year + fractional_year

    def calculate_profitability_index(self, cash_flows: pd.DataFrame) -> float:
        capital_costs = self.calculate_capital_costs()
        initial_investment = capital_costs['tci']

        future_cash_flows = cash_flows[cash_flows['Year'] > 0]['Cash Flow']
        years = cash_flows[cash_flows['Year'] > 0]['Year']

        pv_future_cash_flows = sum([cf / (1 + self.financial.discount_rate) ** year
                                   for cf, year in zip(future_cash_flows, years)])

        return pv_future_cash_flows / initial_investment

    def calculate_roi(self, cash_flows: pd.DataFrame) -> float:
        # Calculate average annual net income after tax for operating years
        operating_years_cash_flows = cash_flows[cash_flows['Year'] > 0]
        total_net_income = operating_years_cash_flows['Net Income'].sum()
        average_annual_net_income = total_net_income / self.financial.project_lifespan

        # Get the initial investment from Year 0 CAPEX
        initial_investment = abs(cash_flows['CAPEX'].iloc[0])

        if initial_investment == 0:
            return np.nan # Avoid division by zero

        return (average_annual_net_income / initial_investment) * 100

    def sensitivity_analysis(self, variable_name: str, base_value: float,
                           variations: List[float]) -> pd.DataFrame:
        results = []

        for variation in variations:
            temp_model = TigernutOilTEA()

            if variable_name == 'raw_material_cost':
                temp_model.operating.raw_material_cost = base_value * (1 + variation)
            elif variable_name == 'oil_price':
                temp_model.financial.oil_price = base_value * (1 + variation)
            elif variable_name == 'cake_price':
                temp_model.financial.cake_price = base_value * (1 + variation)
            elif variable_name == 'daily_capacity':
                temp_model.process.daily_capacity = base_value * (1 + variation)

            cash_flows = temp_model.calculate_cash_flows()
            npv = temp_model.calculate_npv(cash_flows)
            irr = temp_model.calculate_irr(cash_flows)
            payback = temp_model.calculate_payback_period(cash_flows)
            pi = temp_model.calculate_profitability_index(cash_flows)

            results.append({
                'Variation': variation * 100,
                'NPV': npv,
                'IRR': irr * 100,
                'Payback Period': payback,
                'Profitability Index': pi
            })

        return pd.DataFrame(results)

    def monte_carlo_simulation(self, iterations: int = 10000) -> Dict:
        npv_results = []
        irr_results = []
        payback_results = []
        pi_results = []

        for _ in range(iterations):
            temp_model = TigernutOilTEA()

            temp_model.operating.raw_material_cost = np.random.triangular(4.0, 5.0, 6.0)
            temp_model.financial.oil_price = np.random.triangular(25.0, 30.0, 35.0)
            temp_model.financial.cake_price = np.random.triangular(1.5, 1.8, 2.1)
            temp_model.process.daily_capacity = np.random.triangular(1800, 2000, 2200)

            cash_flows = temp_model.calculate_cash_flows()
            npv = temp_model.calculate_npv(cash_flows)
            irr = temp_model.calculate_irr(cash_flows)
            payback = temp_model.calculate_payback_period(cash_flows)
            pi = temp_model.calculate_profitability_index(cash_flows)

            npv_results.append(npv)
            irr_results.append(irr)
            payback_results.append(payback)
            pi_results.append(pi)

        return {
            'npv': {
                'mean': np.mean(npv_results),
                'std': np.std(npv_results),
                'min': np.min(npv_results),
                'max': np.max(npv_results),
                'percentile_5': np.percentile(npv_results, 5),
                'percentile_95': np.percentile(npv_results, 95)
            },
            'irr': {
                'mean': np.mean(irr_results) * 100,
                'std': np.std(irr_results) * 100,
                'min': np.min(irr_results) * 100,
                'max': np.max(irr_results) * 100,
                'percentile_5': np.percentile(irr_results, 5) * 100,
                'percentile_95': np.percentile(irr_results, 95) * 100
            },
            'payback': {
                'mean': np.mean(payback_results),
                'std': np.std(payback_results),
                'min': np.min(payback_results),
                'max': np.max(payback_results),
                'percentile_5': np.percentile(payback_results, 5),
                'percentile_95': np.percentile(payback_results, 95)
            },
            'pi': {
                'mean': np.mean(pi_results),
                'std': np.std(pi_results),
                'min': np.min(pi_results),
                'max': np.max(pi_results),
                'percentile_5': np.percentile(pi_results, 5),
                'percentile_95': np.percentile(pi_results, 95)
            },
            'raw_data': {
                'npv': npv_results,
                'irr': irr_results,
                'payback': payback_results,
                'pi': pi_results
            }
        }

    def scenario_analysis(self) -> pd.DataFrame:
        scenarios = []

        optimistic = TigernutOilTEA()
        optimistic.operating.raw_material_cost = 4.0
        optimistic.financial.oil_price = 35.0
        optimistic.financial.cake_price = 2.1
        optimistic.process.daily_capacity = 2200

        cash_flows_opt = optimistic.calculate_cash_flows()
        scenarios.append({
            'Scenario': 'Optimistic',
            'NPV': optimistic.calculate_npv(cash_flows_opt),
            'IRR': optimistic.calculate_irr(cash_flows_opt) * 100,
            'Payback Period': optimistic.calculate_payback_period(cash_flows_opt),
            'Profitability Index': optimistic.calculate_profitability_index(cash_flows_opt),
            'ROI': optimistic.calculate_roi(cash_flows_opt)
        })

        base = TigernutOilTEA()
        cash_flows_base = base.calculate_cash_flows()
        scenarios.append({
            'Scenario': 'Base Case',
            'NPV': base.calculate_npv(cash_flows_base),
            'IRR': base.calculate_irr(cash_flows_base) * 100,
            'Payback Period': base.calculate_payback_period(cash_flows_base),
            'Profitability Index': base.calculate_profitability_index(cash_flows_base),
            'ROI': base.calculate_roi(cash_flows_base)
        })

        pessimistic = TigernutOilTEA()
        pessimistic.operating.raw_material_cost = 6.0
        pessimistic.financial.oil_price = 25.0
        pessimistic.financial.cake_price = 1.5
        pessimistic.process.daily_capacity = 1800

        cash_flows_pess = pessimistic.calculate_cash_flows()
        scenarios.append({
            'Scenario': 'Pessimistic',
            'NPV': pessimistic.calculate_npv(cash_flows_pess),
            'IRR': pessimistic.calculate_irr(cash_flows_pess) * 100,
            'Payback Period': pessimistic.calculate_payback_period(cash_flows_pess),
            'Profitability Index': pessimistic.calculate_profitability_index(cash_flows_pess),
            'ROI': pessimistic.calculate_roi(cash_flows_pess)
        })

        return pd.DataFrame(scenarios)

    def break_even_analysis(self) -> Dict:
        def find_break_even_capacity():
            low, high = 0.1, 1.0
            tolerance = 0.001

            while high - low > tolerance:
                mid = (low + high) / 2
                temp_model = TigernutOilTEA()
                temp_model.financial.capacity_utilization_year1 = mid
                temp_model.financial.capacity_utilization_year3plus = mid

                cash_flows = temp_model.calculate_cash_flows()
                npv = temp_model.calculate_npv(cash_flows)

                if npv > 0:
                    high = mid
                else:
                    low = mid

            return mid * 100

        def find_break_even_oil_price():
            low, high = 10.0, 50.0
            tolerance = 0.01

            while high - low > tolerance:
                mid = (low + high) / 2
                temp_model = TigernutOilTEA()
                temp_model.financial.oil_price = mid

                cash_flows = temp_model.calculate_cash_flows()
                npv = temp_model.calculate_npv(cash_flows)

                if npv > 0:
                    high = mid
                else:
                    low = mid

            return mid

        return {
            'break_even_capacity_utilization': find_break_even_capacity(),
            'break_even_oil_price': find_break_even_oil_price()
        }

    def generate_report(self) -> Dict:
        mass_balance = self.calculate_mass_balance()
        capital_costs = self.calculate_capital_costs()
        operating_costs = self.calculate_operating_costs(mass_balance)
        cash_flows = self.calculate_cash_flows()

        npv = self.calculate_npv(cash_flows)
        irr = self.calculate_irr(cash_flows)
        payback = self.calculate_payback_period(cash_flows)
        pi = self.calculate_profitability_index(cash_flows)
        roi = self.calculate_roi(cash_flows)

        variations = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
        sensitivity_raw_material = self.sensitivity_analysis('raw_material_cost',
                                                            self.operating.raw_material_cost,
                                                            variations)
        sensitivity_oil_price = self.sensitivity_analysis('oil_price',
                                                         self.financial.oil_price,
                                                         variations)
        sensitivity_cake_price = self.sensitivity_analysis('cake_price',
                                                         self.financial.cake_price,
                                                         variations)

        monte_carlo_results = self.monte_carlo_simulation(10000)
        scenario_results = self.scenario_analysis()
        break_even = self.break_even_analysis()

        return {
            'mass_balance': mass_balance,
            'capital_costs': capital_costs,
            'operating_costs': operating_costs,
            'cash_flows': cash_flows,
            'financial_metrics': {
                'NPV': npv,
                'IRR': irr * 100,
                'Payback Period': payback,
                'Profitability Index': pi,
                'ROI': roi
            },
            'sensitivity_analysis': {
                'raw_material': sensitivity_raw_material,
                'oil_price': sensitivity_oil_price,
                'cake_price': sensitivity_cake_price
            },
            'monte_carlo': monte_carlo_results,
            'scenarios': scenario_results,
            'break_even': break_even
        }

def visualize_results(model: TigernutOilTEA):
    results = model.generate_report()

    fig, axes = plt.subplots(3, 2, figsize=(15, 15)) # Adjusted for 6 plots

    # Annual Cash Flows
    cash_flows = results['cash_flows']
    axes[0, 0].bar(cash_flows['Year'], cash_flows['Cash Flow'])
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Cash Flow ($)')
    axes[0, 0].set_title('Annual Cash Flows')
    axes[0, 0].grid(False)

    # Cumulative Cash Flow
    cumulative_cf = cash_flows['Cash Flow'].cumsum()
    axes[0, 1].plot(cash_flows['Year'], cumulative_cf, marker='o')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Cumulative Cash Flow ($)')
    axes[0, 1].set_title('Cumulative Cash Flow')
    axes[0, 1].grid(False)

    # Sensitivity Analysis - NPV
    variations = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    raw_material_npv = results['sensitivity_analysis']['raw_material']['NPV'].values
    oil_price_npv = results['sensitivity_analysis']['oil_price']['NPV'].values
    cake_price_npv = results['sensitivity_analysis']['cake_price']['NPV'].values

    axes[1, 0].plot(variations, raw_material_npv, marker='o', label='Raw Material Cost')
    axes[1, 0].plot(variations, oil_price_npv, marker='s', label='Oil Price')
    axes[1, 0].plot(variations, cake_price_npv, marker='^', label='Cake Price')
    axes[1, 0].axhline(y=results['financial_metrics']['NPV'], color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Variation (%)')
    axes[1, 0].set_ylabel('NPV ($)')
    axes[1, 0].set_title('Sensitivity Analysis - NPV')
    axes[1, 0].legend()
    axes[1, 0].grid(False)

    # Monte Carlo Simulation - NPV Distribution
    mc_npv = results['monte_carlo']['raw_data']['npv']
    axes[1, 1].hist(mc_npv, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=np.mean(mc_npv), color='r', linestyle='--', label=f'Mean: ${np.mean(mc_npv):,.0f}')
    axes[1, 1].set_xlabel('NPV ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Monte Carlo Simulation - NPV Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(False)

    # Monte Carlo Simulation - IRR Distribution
    mc_irr = [x * 100 for x in results['monte_carlo']['raw_data']['irr'] if not np.isinf(x) and not np.isnan(x)] # Convert to percentage and handle inf/nan
    axes[2, 0].hist(mc_irr, bins=50, edgecolor='black', alpha=0.7)
    axes[2, 0].axvline(x=np.mean(mc_irr), color='r', linestyle='--', label=f'Mean: {np.mean(mc_irr):.2f}%')
    axes[2, 0].set_xlabel('IRR (%)')
    axes[2, 0].set_ylabel('Frequency')
    axes[2, 0].set_title('Monte Carlo Simulation - IRR Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(False)

    # Monte Carlo Simulation - Payback Period Distribution
    mc_payback = [x for x in results['monte_carlo']['raw_data']['payback'] if not np.isinf(x) and not np.isnan(x)] # Handle inf/nan
    axes[2, 1].hist(mc_payback, bins=50, edgecolor='black', alpha=0.7)
    axes[2, 1].axvline(x=np.mean(mc_payback), color='r', linestyle='--', label=f'Mean: {np.mean(mc_payback):.2f} years')
    axes[2, 1].set_xlabel('Payback Period (years)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].set_title('Monte Carlo Simulation - Payback Period Distribution')
    axes[2, 1].legend()
    axes[2, 1].grid(False)

    plt.tight_layout()
    plt.savefig('tigernut_tea_analysis_montecarlo.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Scenario Analysis and Operating Cost Breakdown plots (keep original layout)
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

    scenarios = results['scenarios']
    x_pos = np.arange(len(scenarios))
    axes2[0].bar(x_pos, scenarios['NPV'].values)
    axes2[0].set_xticks(x_pos)
    axes2[0].set_xticklabels(scenarios['Scenario'].values, rotation=45)
    axes2[0].set_ylabel('NPV ($)')
    axes2[0].set_title('Scenario Analysis - NPV')
    axes2[0].grid(False)

    cost_breakdown = {
        'Raw Materials': results['operating_costs']['raw_material_cost'],
        'Utilities': results['operating_costs']['utility_cost'],
        'Labor': results['operating_costs']['total_labor_cost'],
        'Lab/QC/QA': results['operating_costs']['lab_qc_qa_cost'],
        'Maintenance': results['operating_costs']['maintenance_cost'],
        'Overhead': results['operating_costs']['overhead']
    }

    axes2[1].pie(cost_breakdown.values(), labels=cost_breakdown.keys(), autopct='%1.1f%%')
    axes2[1].set_title('Operating Cost Breakdown')
    axes2[1].grid(False)

    plt.tight_layout()
    plt.savefig('tigernut_tea_analysis_scenarios_costs.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary(model: TigernutOilTEA):
    results = model.generate_report()

    print("="*60)
    print("TIGERNUT OIL PROCESSING - TECHNO-ECONOMIC ANALYSIS SUMMARY")
    print("="*60)

    print("\nMASS BALANCE:")
    print(f"  Annual Input: {results['mass_balance']['annual_input']:,.0f} kg")
    print(f"  Annual Oil Production: {results['mass_balance']['annual_oil']:,.0f} kg")
    print(f"  Annual Cake Production: {results['mass_balance']['annual_cake']:,.0f} kg")
    print(f"  Annual Losses: {results['mass_balance']['annual_losses']:,.0f} kg")
    print(f"  Oil Yield: {results['mass_balance']['oil_yield_percentage']:.2f}%")
    print(f"  Cake Yield: {results['mass_balance']['cake_yield_percentage']:.2f}%")

    print("\nCAPITAL INVESTMENT:")
    print(f"  Total Equipment Cost: ${results['capital_costs']['tepc']:,.0f}")
    print(f"  Fixed Capital Investment: ${results['capital_costs']['fci']:,.0f}")
    print(f"  Total Capital Investment: ${results['capital_costs']['tci']:,.0f}")

    print("\nANNUAL OPERATING COSTS:")
    print(f"  Raw Materials: ${results['operating_costs']['raw_material_cost']:,.0f}")
    print(f"  Utilities: ${results['operating_costs']['utility_cost']:,.0f}")
    print(f"  Labor: ${results['operating_costs']['total_labor_cost']:,.0f}")
    print(f"  Lab/QC/QA: ${results['operating_costs']['lab_qc_qa_cost']:,.0f}")
    print(f"  Maintenance: ${results['operating_costs']['maintenance_cost']:,.0f}")
    print(f"  Overhead: ${results['operating_costs']['overhead']:,.0f}")
    print(f"  Total Operating Cost: ${results['operating_costs']['total_operating_cost']:,.0f}")

    print("\nFINANCIAL METRICS:")
    print(f"  Net Present Value (NPV): ${results['financial_metrics']['NPV']:,.0f}")
    print(f"  Internal Rate of Return (IRR): {results['financial_metrics']['IRR']:.2f}%")
    print(f"  Payback Period: {results['financial_metrics']['Payback Period']:.2f} years")
    print(f"  Profitability Index: {results['financial_metrics']['Profitability Index']:.2f}")
    print(f"  Return on Investment (ROI): {results['financial_metrics']['ROI']:.2f}%")

    print("\nANNUAL REVENUE:")
    for _, row in results['cash_flows'].iterrows():
        if row['Year'] > 0:
            print(f"  Year {int(row['Year'])}:")
            print(f"    Total Revenue: ${row['Total Revenue']:,.0f}")
            print(f"    Oil Revenue: ${row['Oil Revenue']:,.0f}")
            print(f"    Cake Revenue: ${row['Cake Revenue']:,.0f}")

    print("\nMONTE CARLO SIMULATION (10,000 iterations):")
    print(f"  NPV Mean: ${results['monte_carlo']['npv']['mean']:,.0f} (Std Dev: ${results['monte_carlo']['npv']['std']:,.0f})")
    print(f"  NPV 90% Confidence Interval: ${results['monte_carlo']['npv']['percentile_5']:,.0f} - ${results['monte_carlo']['npv']['percentile_95']:,.0f}")
    print(f"  IRR Mean: {results['monte_carlo']['irr']['mean']:.2f}% (Std Dev: {results['monte_carlo']['irr']['std']:.2f}%)")
    print(f"  Payback Mean: {results['monte_carlo']['payback']['mean']:.2f} years (Std Dev: {results['monte_carlo']['payback']['std']:.2f} years)")
    print(f"  Profitability Index Mean: {results['monte_carlo']['pi']['mean']:.2f} (Std Dev: {results['monte_carlo']['pi']['std']:.2f})")

    print("\nBREAK-EVEN ANALYSIS:")
    print(f"  Break-even Capacity Utilization: {results['break_even']['break_even_capacity_utilization']:.1f}%")
    print(f"  Break-even Oil Price: ${results['break_even']['break_even_oil_price']:.2f}/kg")

    print("\nSCENARIO ANALYSIS:")
    for _, row in results['scenarios'].iterrows():
        print(f"  {row['Scenario']}:")
        print(f"    NPV: ${row['NPV']:,.0f}")
        print(f"    IRR: {row['IRR']:.2f}%")
        print(f"    ROI: {row['ROI']:.2f}%")

if __name__ == "__main__":
    model = TigernutOilTEA()

    print_summary(model)

    fig = visualize_results(model)

    results = model.generate_report()
    cash_flows_df = results['cash_flows']
    cash_flows_df.to_csv('cash_flows.csv', index=False)

    sensitivity_df = pd.concat([
        results['sensitivity_analysis']['raw_material'].assign(Variable='Raw Material Cost'),
        results['sensitivity_analysis']['oil_price'].assign(Variable='Oil Price'),
        results['sensitivity_analysis']['cake_price'].assign(Variable='Cake Price')
    ])
    sensitivity_df.to_csv('sensitivity_analysis.csv', index=False)


