<!-- Doctor Dashboard Component -->
<template>
  <div class="doctor-dashboard">
    <header class="dashboard-header">
      <h1>BEACON Medical Dashboard</h1>
      <div class="user-info">
        <span>{{ doctorName }}</span>
        <button @click="logout">Logout</button>
      </div>
    </header>

    <div class="dashboard-content">
      <!-- Patient List Section -->
      <section class="patient-list">
        <h2>Patients</h2>
        <div class="search-bar">
          <input v-model="searchQuery" placeholder="Search patients...">
          <button @click="addNewPatient">Add New Patient</button>
        </div>
        
        <table class="patient-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Age</th>
              <th>Cancer Type</th>
              <th>Stage</th>
              <th>Last Visit</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="patient in filteredPatients" :key="patient.id">
              <td>{{ patient.id }}</td>
              <td>{{ patient.name }}</td>
              <td>{{ patient.age }}</td>
              <td>{{ patient.cancerType }}</td>
              <td>{{ patient.stage }}</td>
              <td>{{ formatDate(patient.lastVisit) }}</td>
              <td>
                <button @click="viewPatient(patient)">View</button>
                <button @click="editPatient(patient)">Edit</button>
              </td>
            </tr>
          </tbody>
        </table>
      </section>

      <!-- Analysis Section -->
      <section class="analysis-section">
        <h2>Analysis & Predictions</h2>
        <div class="analysis-cards">
          <div class="analysis-card" v-for="analysis in analysisResults" :key="analysis.id">
            <h3>{{ analysis.title }}</h3>
            <div class="analysis-content">
              <div class="metrics">
                <div v-for="metric in analysis.metrics" :key="metric.name">
                  <span class="metric-name">{{ metric.name }}:</span>
                  <span class="metric-value">{{ metric.value }}</span>
                </div>
              </div>
              <div class="chart-container" v-if="analysis.chart">
                <!-- Chart component will be added here -->
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Treatment Plans Section -->
      <section class="treatment-plans">
        <h2>Treatment Plans</h2>
        <div class="treatment-cards">
          <div class="treatment-card" v-for="plan in treatmentPlans" :key="plan.id">
            <h3>{{ plan.patientName }}</h3>
            <div class="plan-details">
              <p><strong>Cancer Type:</strong> {{ plan.cancerType }}</p>
              <p><strong>Stage:</strong> {{ plan.stage }}</p>
              <p><strong>Current Phase:</strong> {{ plan.currentPhase }}</p>
              <p><strong>Next Review:</strong> {{ formatDate(plan.nextReview) }}</p>
            </div>
            <div class="plan-actions">
              <button @click="viewPlan(plan)">View Details</button>
              <button @click="updatePlan(plan)">Update Plan</button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- Modals -->
    <div v-if="showPatientModal" class="modal">
      <div class="modal-content">
        <h2>Patient Details</h2>
        <!-- Patient details form -->
      </div>
    </div>

    <div v-if="showTreatmentModal" class="modal">
      <div class="modal-content">
        <h2>Treatment Plan</h2>
        <!-- Treatment plan form -->
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DoctorDashboard',
  
  data() {
    return {
      doctorName: 'Dr. Smith',
      searchQuery: '',
      patients: [],
      analysisResults: [],
      treatmentPlans: [],
      showPatientModal: false,
      showTreatmentModal: false,
      selectedPatient: null,
      selectedPlan: null
    }
  },

  computed: {
    filteredPatients() {
      return this.patients.filter(patient => {
        const searchLower = this.searchQuery.toLowerCase()
        return (
          patient.name.toLowerCase().includes(searchLower) ||
          patient.id.toLowerCase().includes(searchLower) ||
          patient.cancerType.toLowerCase().includes(searchLower)
        )
      })
    }
  },

  methods: {
    async fetchPatients() {
      try {
        const response = await this.$http.get('/api/patients')
        this.patients = response.data
      } catch (error) {
        console.error('Error fetching patients:', error)
        this.$toast.error('Failed to load patients')
      }
    },

    async fetchAnalysis() {
      try {
        const response = await this.$http.get('/api/analysis')
        this.analysisResults = response.data
      } catch (error) {
        console.error('Error fetching analysis:', error)
        this.$toast.error('Failed to load analysis results')
      }
    },

    async fetchTreatmentPlans() {
      try {
        const response = await this.$http.get('/api/treatment-plans')
        this.treatmentPlans = response.data
      } catch (error) {
        console.error('Error fetching treatment plans:', error)
        this.$toast.error('Failed to load treatment plans')
      }
    },

    formatDate(date) {
      return new Date(date).toLocaleDateString()
    },

    addNewPatient() {
      this.selectedPatient = null
      this.showPatientModal = true
    },

    viewPatient(patient) {
      this.selectedPatient = patient
      this.showPatientModal = true
    },

    editPatient(patient) {
      this.selectedPatient = { ...patient }
      this.showPatientModal = true
    },

    viewPlan(plan) {
      this.selectedPlan = plan
      this.showTreatmentModal = true
    },

    updatePlan(plan) {
      this.selectedPlan = { ...plan }
      this.showTreatmentModal = true
    },

    async savePatient(patientData) {
      try {
        if (patientData.id) {
          await this.$http.put(`/api/patients/${patientData.id}`, patientData)
        } else {
          await this.$http.post('/api/patients', patientData)
        }
        this.showPatientModal = false
        this.fetchPatients()
        this.$toast.success('Patient data saved successfully')
      } catch (error) {
        console.error('Error saving patient:', error)
        this.$toast.error('Failed to save patient data')
      }
    },

    async saveTreatmentPlan(planData) {
      try {
        if (planData.id) {
          await this.$http.put(`/api/treatment-plans/${planData.id}`, planData)
        } else {
          await this.$http.post('/api/treatment-plans', planData)
        }
        this.showTreatmentModal = false
        this.fetchTreatmentPlans()
        this.$toast.success('Treatment plan saved successfully')
      } catch (error) {
        console.error('Error saving treatment plan:', error)
        this.$toast.error('Failed to save treatment plan')
      }
    },

    logout() {
      this.$store.dispatch('logout')
      this.$router.push('/login')
    }
  },

  created() {
    this.fetchPatients()
    this.fetchAnalysis()
    this.fetchTreatmentPlans()
  }
}
</script>

<style scoped>
.doctor-dashboard {
  padding: 20px;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.dashboard-content {
  display: grid;
  grid-gap: 20px;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}

.patient-list {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.search-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.search-bar input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.patient-table {
  width: 100%;
  border-collapse: collapse;
}

.patient-table th,
.patient-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

.analysis-section {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.analysis-cards {
  display: grid;
  grid-gap: 20px;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.analysis-card {
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.treatment-plans {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.treatment-cards {
  display: grid;
  grid-gap: 20px;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.treatment-card {
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal-content {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  min-width: 500px;
}

button {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  background-color: #4CAF50;
  color: white;
  cursor: pointer;
}

button:hover {
  background-color: #45a049;
}

.metric-name {
  font-weight: bold;
  margin-right: 8px;
}

.chart-container {
  margin-top: 15px;
  height: 200px;
}
</style> 