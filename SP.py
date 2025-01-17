import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ChevronRight, Database, Target, Beaker, LineChart, ArrowRightCircle } from 'lucide-react';

const ResearchProposal = () => {
  const [selectedSection, setSelectedSection] = useState('overview');

  const TopBar = () => (
    <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-6 rounded-t-lg">
      <h1 className="text-3xl font-bold mb-2">Environmental-Economic Sensitivity Analysis</h1>
      <p className="text-blue-100">US Regional Semiconductor Supply Chain Dynamics</p>
    </div>
  );

  const SideNav = () => (
    <div className="flex flex-col gap-2 w-64 p-4 bg-gray-50 rounded-lg">
      {['overview', 'methodology', 'regions', 'data', 'equations', 'results'].map((section) => (
        <button
          key={section}
          className={`flex items-center gap-2 p-3 rounded-lg text-left transition-all ${
            selectedSection === section 
              ? 'bg-blue-600 text-white shadow-lg transform scale-105' 
              : 'bg-white hover:bg-gray-100'
          }`}
          onClick={() => setSelectedSection(section)}
        >
          <ChevronRight className={`w-4 h-4 ${selectedSection === section ? 'text-white' : 'text-blue-600'}`} />
          {section.charAt(0).toUpperCase() + section.slice(1)}
        </button>
      ))}
    </div>
  );

  const RegionCard = ({ name, characteristics }) => (
    <div className="bg-white p-4 rounded-lg shadow-md hover:shadow-lg transition-shadow">
      <h3 className="text-lg font-semibold text-blue-800 mb-2">{name}</h3>
      <ul className="space-y-2">
        {characteristics.map((char, idx) => (
          <li key={idx} className="flex items-start gap-2">
            <ArrowRightCircle className="w-4 h-4 mt-1 text-blue-500" />
            {char}
          </li>
        ))}
      </ul>
    </div>
  );

  const renderContent = () => {
    switch(selectedSection) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-2xl font-bold text-blue-800 mb-4">Research Overview</h2>
              <p className="text-gray-700 leading-relaxed">
                This cutting-edge research investigates the complex dynamics between environmental sustainability 
                and semiconductor supply chain resilience across key US manufacturing regions. Using advanced 
                nonlinear dynamical systems analysis, we model the intricate relationships between water 
                availability, energy transitions, and environmental regulations.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-blue-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Key Innovations</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Nonlinear coupling of environmental-economic factors</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Regional sensitivity analysis framework</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Target className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Critical threshold identification methods</span>
                  </li>
                </ul>
              </div>

              <div className="bg-green-50 p-6 rounded-lg">
                <h3 className="text-xl font-semibold text-green-800 mb-3">Expected Impact</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <span>Enhanced supply chain resilience strategies</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <span>Regional policy recommendations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <span>Sustainability-oriented manufacturing practices</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        );

      case 'regions':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-blue-800">Regional Analysis</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <RegionCard 
                name="Southwest Region (AZ, NM)"
                characteristics={[
                  "Water scarcity challenges",
                  "High solar energy potential",
                  "Major players: Intel, TSMC"
                ]}
              />
              <RegionCard 
                name="Pacific Northwest (OR, WA)"
                characteristics={[
                  "Hydroelectric power availability",
                  "Stable water supply",
                  "Major player: Intel"
                ]}
              />
              <RegionCard 
                name="Texas Region"
                characteristics={[
                  "Independent power grid (ERCOT)",
                  "Mixed energy sources",
                  "Major players: Samsung, TI"
                ]}
              />
              <RegionCard 
                name="Northeast Corridor (NY)"
                characteristics={[
                  "Stable water resources",
                  "Strict environmental regulations",
                  "Major player: GlobalFoundries"
                ]}
              />
            </div>
          </div>
        );

      case 'data':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-blue-800 mb-4">Data Requirements & Accessibility</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-b from-green-50 to-white p-6 rounded-lg shadow-md">
                <h3 className="font-semibold flex items-center gap-2 text-green-800 mb-4">
                  <Database className="w-5 h-5 text-green-600" />
                  Easily Accessible Data
                </h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Regional Energy Consumption</p>
                      <p className="text-sm text-gray-600">Source: Department of Energy (DOE)</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Water Usage Permits</p>
                      <p className="text-sm text-gray-600">Source: State Environmental Agencies</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Environmental Compliance Records</p>
                      <p className="text-sm text-gray-600">Source: EPA Database</p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-gradient-to-b from-yellow-50 to-white p-6 rounded-lg shadow-md">
                <h3 className="font-semibold flex items-center gap-2 text-yellow-800 mb-4">
                  <Database className="w-5 h-5 text-yellow-600" />
                  Moderately Difficult Data
                </h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-yellow-600" />
                    <div>
                      <p className="font-semibold">Facility Energy Usage</p>
                      <p className="text-sm text-gray-600">Source: Company Reports</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-yellow-600" />
                    <div>
                      <p className="font-semibold">Water Recycling Rates</p>
                      <p className="text-sm text-gray-600">Source: Industry Surveys</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-yellow-600" />
                    <div>
                      <p className="font-semibold">Production Capacity</p>
                      <p className="text-sm text-gray-600">Source: Industry Reports</p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-gradient-to-b from-red-50 to-white p-6 rounded-lg shadow-md">
                <h3 className="font-semibold flex items-center gap-2 text-red-800 mb-4">
                  <Database className="w-5 h-5 text-red-600" />
                  Challenging Data
                </h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-red-600" />
                    <div>
                      <p className="font-semibold">Efficiency Metrics</p>
                      <p className="text-sm text-gray-600">Source: Proprietary Data</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-red-600" />
                    <div>
                      <p className="font-semibold">Production Costs</p>
                      <p className="text-sm text-gray-600">Source: Internal Records</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-red-600" />
                    <div>
                      <p className="font-semibold">Environmental Targets</p>
                      <p className="text-sm text-gray-600">Source: Corporate Plans</p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        );

      case 'methodology':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-blue-800 mb-4">Research Methodology</h2>
            
            <div className="bg-white p-6 rounded-lg shadow-md mb-6">
              <h3 className="text-xl font-semibold text-blue-800 mb-4">System Relationships</h3>
              <div className="bg-blue-50 p-4 rounded-lg">
                <img src="https://mermaid.ink/svg/pako:eNqNVE1v2zAM_SuETh2QFPDHbKd2l2LHYcBQoEOBXQxZYmyhsmRIcpag6H8fZcVJ3BbdepIoPpLvPVK6Q7UVHHmBV1t0pZYWDIOgNpK9K_LMtNELqX_W0lSw9F4YwZXWsNkHa1EAO31RjS4rqBsHX1SDHUTwR5XWQG2hAm05_oGNbjwE-vXBGOl2J8f-K_QJ0I_-A2c5CNYBNByVRFwVypTDI-6EQdPZllD8kDd4lT4YZb_Mj-Kre2mDVahzPu-YTmfHF47m0-hs8XEyn0Uw_YFdJa1x0FQ6d9jHfJwvZvPZdBGmoQeKBg6tM1ZXEVzyQpemdm2jvVPRzfnrUVfrJ8_J1awggl-gK24i-HRzG8Eq18aUaGBtZV0BVEDxr-1r9sC7DkIo-Uarpr94Zdy3PZEILv4tTQkl5xDkXTLgpxX4ZWfBmfB8NORQqXzL20qhLXkJtVWtm5QXDVaFthUHqKFlpTXgHQFxbfJDVqmMQ2d7qT1qbhVP_uDJaBSGvJvYJNrGUYRBSK9dZqWl1i-O4Dd4z_PK1C3lEXwbz-KPcXx3n97eX8fjuxh-Jq32pbRRPptKqxzCJvHBJ2sJfpO_IEQEDy9_5cBF8HkSD75GcJvQ5CabSS4F_OEm58nkNI1RaK3x0uCw2KrIQ-Yj9wB8RB5_0yZrW39JvWWlSGYp7K1eJ4s0S9MkWWVZnK6SNM2yLFuv4jSJ09U6WedxtobvKOy3LWdw2F1BzSQN1LVDAZ4oL0zRIFcFWvBN7sC6UqcLBfQ5ghq54AV4Jrw9BmLDSVzD0PAEDb1vRSE1r7j4L5t7sBiMRuvRaK_4zoGQ3WK5YH9pT3YmQ9tBKQv_8VDlJCYVS1GcXsqeCW91oet9oaU5kcm7bRoP3W_xHzBYREI" alt="System Relationships" className="w-full" />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Analysis Methods</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Lyapunov Stability Analysis</p>
                      <p className="text-sm text-gray-600">Examining system stability near equilibrium points</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Bifurcation Analysis</p>
                      <p className="text-sm text-gray-600">Identifying critical parameter thresholds</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Sensitivity Analysis</p>
                      <p className="text-sm text-gray-600">Parameter impact assessment</p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Implementation Steps</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Data Collection & Validation</p>
                      <p className="text-sm text-gray-600">Regional environmental and production data</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Model Calibration</p>
                      <p className="text-sm text-gray-600">Parameter estimation and validation</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRightCircle className="w-4 h-4 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Regional Analysis</p>
                      <p className="text-sm text-gray-600">Comparative regional studies</p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        );

      case 'equations':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-blue-800">System Equations</h2>
            <div className="bg-blue-50 p-6 rounded-lg shadow-md">
              <h3 className="text-xl font-semibold text-blue-800 mb-4">Nonlinear Dynamical System</h3>
              <div className="font-mono text-sm space-y-3 bg-white p-4 rounded-lg">
                <p>dP/dt = μ₁M(t)E(t)W(t)(1 - P/K) - δ₁D(t)P²</p>
                <p>dW/dt = α₂P(t)(1 - W/Wmax) - β₂R(t)W² - δ₂T(t)</p>
                <p>dE/dt = [α₁P(t) + β₁M(t)](1 - E/Emax) - γ₁R(t)E²</p>
                <p>dC/dt = λ₁(C*(t) - C(t))³ + λ₂E(t)W(t) - λ₄P(t)²</p>
                <p>dR/dt = σ₁C(t)² + σ₂(E(t)/P(t))³ + σ₃(W(t)/P(t))³ - σ₄R(t)²</p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Stability Analysis</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Lyapunov stability analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Bifurcation analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Beaker className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Phase space analysis</span>
                  </li>
                </ul>
              </div>
              <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800 mb-3">Parameters</h3>
                <ul className="space-y-3">
                  <li className="flex items-start gap-2">
                    <Database className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Environmental coupling coefficients</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Database className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Production efficiency factors</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <Database className="w-5 h-5 mt-1 text-blue-600" />
                    <span>Resource utilization rates</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        );

      case 'results':
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-blue-800 mb-4">Expected Results & Impact</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-gradient-to-br from-blue-50 via-white to-blue-50 p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-blue-800 mb-4">Research Outcomes</h3>
                <ul className="space-y-4">
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Regional Stability Maps</p>
                      <p className="text-sm text-gray-600">Identification of stability boundaries and critical thresholds for each region</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Sensitivity Metrics</p>
                      <p className="text-sm text-gray-600">Quantitative measures of system response to parameter variations</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-blue-600" />
                    <div>
                      <p className="font-semibold">Risk Assessment Framework</p>
                      <p className="text-sm text-gray-600">Comprehensive evaluation of environmental risks to supply chain stability</p>
                    </div>
                  </li>
                </ul>
              </div>

              <div className="bg-gradient-to-br from-green-50 via-white to-green-50 p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold text-green-800 mb-4">Expected Impact</h3>
                <ul className="space-y-4">
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Policy Recommendations</p>
                      <p className="text-sm text-gray-600">Evidence-based guidance for regional policy makers</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Industry Guidelines</p>
                      <p className="text-sm text-gray-600">Best practices for environmental sustainability</p>
                    </div>
                  </li>
                  <li className="flex items-start gap-3">
                    <LineChart className="w-5 h-5 mt-1 text-green-600" />
                    <div>
                      <p className="font-semibold">Resilience Strategies</p>
                      <p className="text-sm text-gray-600">Actionable plans for supply chain strengthening</p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>

            <div className="bg-gradient-to-br from-purple-50 via-white to-purple-50 p-6 rounded-lg shadow-md mt-6">
              <h3 className="text-xl font-semibold text-purple-800 mb-4">Future Applications</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-4 rounded-lg">
                  <p className="font-semibold text-purple-700">Model Extension</p>
                  <p className="text-sm text-gray-600">Adaptable to other manufacturing sectors</p>
                </div>
                <div className="bg-white p-4 rounded-lg">
                  <p className="font-semibold text-purple-700">Decision Support</p>
                  <p className="text-sm text-gray-600">Interactive planning tools</p>
                </div>
                <div className="bg-white p-4 rounded-lg">
                  <p className="font-semibold text-purple-700">Risk Management</p>
                  <p className="text-sm text-gray-600">Early warning system development</p>
                </div>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <TopBar />
      <div className="max-w-7xl mx-auto p-6">
        <div className="flex gap-6">
          <SideNav />
          <div className="flex-1 bg-white p-6 rounded-lg shadow-md">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResearchProposal;