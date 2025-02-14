<!-- Patient Portal Component -->
<template>
  <div class="patient-portal">
    <header class="portal-header">
      <h1>BEACON Patient Portal</h1>
      <div class="user-info">
        <span>{{ patientName }}</span>
        <button @click="logout">Logout</button>
      </div>
    </header>

    <div class="portal-content">
      <!-- Health Summary Section -->
      <section class="health-summary">
        <h2>Health Summary</h2>
        <div class="summary-cards">
          <div class="summary-card">
            <h3>Diagnosis</h3>
            <p><strong>Cancer Type:</strong> {{ diagnosis.cancerType }}</p>
            <p><strong>Stage:</strong> {{ diagnosis.stage }}</p>
            <p><strong>Diagnosed Date:</strong> {{ formatDate(diagnosis.diagnosedDate) }}</p>
          </div>

          <div class="summary-card">
            <h3>Current Status</h3>
            <p><strong>Treatment Phase:</strong> {{ currentStatus.phase }}</p>
            <p><strong>Last Updated:</strong> {{ formatDate(currentStatus.lastUpdated) }}</p>
            <p><strong>Next Appointment:</strong> {{ formatDate(currentStatus.nextAppointment) }}</p>
          </div>

          <div class="summary-card">
            <h3>Vital Signs</h3>
            <div class="vitals-grid">
              <div v-for="vital in vitals" :key="vital.name">
                <span class="vital-name">{{ vital.name }}:</span>
                <span class="vital-value">{{ vital.value }} {{ vital.unit }}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Treatment Plan Section -->
      <section class="treatment-plan">
        <h2>Treatment Plan</h2>
        <div class="timeline">
          <div class="timeline-item" v-for="phase in treatmentPlan" :key="phase.id"
               :class="{ 'completed': phase.completed, 'current': phase.current }">
            <div class="timeline-marker"></div>
            <div class="timeline-content">
              <h3>{{ phase.name }}</h3>
              <p>{{ phase.description }}</p>
              <p><strong>Duration:</strong> {{ phase.duration }}</p>
              <p><strong>Status:</strong> {{ phase.status }}</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Symptoms & Side Effects Section -->
      <section class="symptoms-tracking">
        <h2>Symptoms & Side Effects</h2>
        <div class="symptoms-form">
          <h3>Record Daily Symptoms</h3>
          <div class="symptom-inputs">
            <div v-for="symptom in symptoms" :key="symptom.name" class="symptom-input">
              <label>{{ symptom.name }}</label>
              <div class="rating-scale">
                <button v-for="n in 5" :key="n"
                        :class="{ active: symptom.rating === n }"
                        @click="rateSymptom(symptom, n)">
                  {{ n }}
                </button>
              </div>
            </div>
          </div>
          <textarea v-model="symptomsNotes" placeholder="Additional notes..."></textarea>
          <button @click="submitSymptoms">Submit Symptoms Report</button>
        </div>

        <div class="symptoms-history">
          <h3>Symptoms History</h3>
          <div class="chart-container">
            <!-- Symptoms trend chart will be added here -->
          </div>
        </div>
      </section>

      <!-- Appointments & Medications Section -->
      <section class="appointments-medications">
        <div class="appointments">
          <h2>Upcoming Appointments</h2>
          <div class="appointment-list">
            <div v-for="appointment in appointments" :key="appointment.id" class="appointment-card">
              <div class="appointment-date">
                <span class="date">{{ formatDate(appointment.date) }}</span>
                <span class="time">{{ formatTime(appointment.time) }}</span>
              </div>
              <div class="appointment-details">
                <h3>{{ appointment.type }}</h3>
                <p><strong>Doctor:</strong> {{ appointment.doctor }}</p>
                <p><strong>Location:</strong> {{ appointment.location }}</p>
              </div>
              <div class="appointment-actions">
                <button @click="rescheduleAppointment(appointment)">Reschedule</button>
                <button @click="cancelAppointment(appointment)">Cancel</button>
              </div>
            </div>
          </div>
        </div>

        <div class="medications">
          <h2>Current Medications</h2>
          <div class="medication-list">
            <div v-for="medication in medications" :key="medication.id" class="medication-card">
              <h3>{{ medication.name }}</h3>
              <p><strong>Dosage:</strong> {{ medication.dosage }}</p>
              <p><strong>Frequency:</strong> {{ medication.frequency }}</p>
              <p><strong>Instructions:</strong> {{ medication.instructions }}</p>
              <div class="medication-schedule">
                <div v-for="time in medication.schedule" :key="time"
                     class="schedule-item"
                     :class="{ completed: isMedicationTaken(medication, time) }">
                  {{ time }}
                </div>
              </div>
              <button @click="recordMedication(medication)">Record Taken</button>
            </div>
          </div>
        </div>
      </section>

      <!-- Resources & Education Section -->
      <section class="resources">
        <h2>Resources & Education</h2>
        <div class="resources-grid">
          <div v-for="resource in educationalResources" :key="resource.id" class="resource-card">
            <h3>{{ resource.title }}</h3>
            <p>{{ resource.description }}</p>
            <div class="resource-links">
              <a :href="resource.url" target="_blank">Read More</a>
              <button @click="downloadResource(resource)">Download</button>
            </div>
          </div>
        </div>
      </section>
    </div>

    <!-- Modals -->
    <div v-if="showAppointmentModal" class="modal">
      <div class="modal-content">
        <h2>Reschedule Appointment</h2>
        <!-- Appointment rescheduling form -->
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PatientPortal',
  
  data() {
    return {
      patientName: 'John Doe',
      diagnosis: {
        cancerType: 'Lung Cancer',
        stage: 'Stage II',
        diagnosedDate: '2024-01-15'
      },
      currentStatus: {
        phase: 'Chemotherapy',
        lastUpdated: '2024-03-01',
        nextAppointment: '2024-03-15'
      },
      vitals: [
        { name: 'Blood Pressure', value: '120/80', unit: 'mmHg' },
        { name: 'Heart Rate', value: '72', unit: 'bpm' },
        { name: 'Temperature', value: '36.8', unit: '°C' },
        { name: 'Weight', value: '70', unit: 'kg' }
      ],
      treatmentPlan: [],
      symptoms: [
        { name: 'Pain', rating: 0 },
        { name: 'Fatigue', rating: 0 },
        { name: 'Nausea', rating: 0 },
        { name: 'Appetite', rating: 0 },
        { name: 'Sleep Quality', rating: 0 }
      ],
      symptomsNotes: '',
      appointments: [],
      medications: [],
      educationalResources: [],
      showAppointmentModal: false,
      selectedAppointment: null
    }
  },

  methods: {
    async fetchPatientData() {
      try {
        const response = await this.$http.get('/api/patient/data')
        const data = response.data
        this.diagnosis = data.diagnosis
        this.currentStatus = data.currentStatus
        this.vitals = data.vitals
        this.treatmentPlan = data.treatmentPlan
      } catch (error) {
        console.error('Error fetching patient data:', error)
        this.$toast.error('Failed to load patient data')
      }
    },

    async fetchAppointments() {
      try {
        const response = await this.$http.get('/api/patient/appointments')
        this.appointments = response.data
      } catch (error) {
        console.error('Error fetching appointments:', error)
        this.$toast.error('Failed to load appointments')
      }
    },

    async fetchMedications() {
      try {
        const response = await this.$http.get('/api/patient/medications')
        this.medications = response.data
      } catch (error) {
        console.error('Error fetching medications:', error)
        this.$toast.error('Failed to load medications')
      }
    },

    async fetchResources() {
      try {
        const response = await this.$http.get('/api/patient/resources')
        this.educationalResources = response.data
      } catch (error) {
        console.error('Error fetching resources:', error)
        this.$toast.error('Failed to load resources')
      }
    },

    formatDate(date) {
      return new Date(date).toLocaleDateString()
    },

    formatTime(time) {
      return new Date(`2000-01-01T${time}`).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    },

    rateSymptom(symptom, rating) {
      symptom.rating = rating
    },

    async submitSymptoms() {
      try {
        await this.$http.post('/api/patient/symptoms', {
          symptoms: this.symptoms,
          notes: this.symptomsNotes,
          date: new Date().toISOString()
        })
        this.$toast.success('Symptoms report submitted successfully')
        this.resetSymptoms()
      } catch (error) {
        console.error('Error submitting symptoms:', error)
        this.$toast.error('Failed to submit symptoms report')
      }
    },

    resetSymptoms() {
      this.symptoms.forEach(symptom => symptom.rating = 0)
      this.symptomsNotes = ''
    },

    async rescheduleAppointment(appointment) {
      this.selectedAppointment = appointment
      this.showAppointmentModal = true
    },

    async cancelAppointment(appointment) {
      try {
        await this.$http.delete(`/api/patient/appointments/${appointment.id}`)
        this.$toast.success('Appointment cancelled successfully')
        this.fetchAppointments()
      } catch (error) {
        console.error('Error cancelling appointment:', error)
        this.$toast.error('Failed to cancel appointment')
      }
    },

    isMedicationTaken(medication, time) {
      // Implementation for checking if medication was taken
      return medication.takenTimes && medication.takenTimes.includes(time)
    },

    async recordMedication(medication) {
      try {
        await this.$http.post('/api/patient/medications/record', {
          medicationId: medication.id,
          timestamp: new Date().toISOString()
        })
        this.$toast.success('Medication recorded successfully')
        this.fetchMedications()
      } catch (error) {
        console.error('Error recording medication:', error)
        this.$toast.error('Failed to record medication')
      }
    },

    async downloadResource(resource) {
      try {
        const response = await this.$http.get(resource.downloadUrl, {
          responseType: 'blob'
        })
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', resource.filename)
        document.body.appendChild(link)
        link.click()
        link.remove()
      } catch (error) {
        console.error('Error downloading resource:', error)
        this.$toast.error('Failed to download resource')
      }
    },

    logout() {
      this.$store.dispatch('logout')
      this.$router.push('/login')
    }
  },

  created() {
    this.fetchPatientData()
    this.fetchAppointments()
    this.fetchMedications()
    this.fetchResources()
  }
}
</script>

<style scoped>
.patient-portal {
  padding: 20px;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.portal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.portal-content {
  display: grid;
  grid-gap: 20px;
}

.health-summary {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.summary-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.summary-card {
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.vitals-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.treatment-plan {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.timeline {
  position: relative;
  padding: 20px 0;
}

.timeline-item {
  position: relative;
  padding-left: 40px;
  margin-bottom: 20px;
}

.timeline-marker {
  position: absolute;
  left: 0;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background-color: #ddd;
}

.timeline-item.completed .timeline-marker {
  background-color: #4CAF50;
}

.timeline-item.current .timeline-marker {
  background-color: #2196F3;
}

.symptoms-tracking {
  background-color: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.symptom-inputs {
  display: grid;
  gap: 15px;
  margin-bottom: 20px;
}

.rating-scale {
  display: flex;
  gap: 10px;
}

.rating-scale button {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: 1px solid #ddd;
  background-color: white;
}

.rating-scale button.active {
  background-color: #4CAF50;
  color: white;
  border-color: #4CAF50;
}

.appointments-medications {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.appointment-card,
.medication-card {
  background-color: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-bottom: 15px;
}

.medication-schedule {
  display: flex;
  gap: 10px;
  margin: 10px 0;
}

.schedule-item {
  padding: 5px 10px;
  border-radius: 15px;
  background-color: #f0f0f0;
}

.schedule-item.completed {
  background-color: #4CAF50;
  color: white;
}

.resources-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.resource-card {
  background-color: white;
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.resource-links {
  display: flex;
  gap: 10px;
  margin-top: 10px;
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

textarea {
  width: 100%;
  min-height: 100px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-bottom: 15px;
}

.chart-container {
  height: 300px;
  margin-top: 20px;
}
</style> 