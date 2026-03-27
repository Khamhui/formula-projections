import Foundation

struct SBRace: Codable {
    let id: String
    let season: Int
    let round: Int
    let name: String
    let circuitId: String
    let circuitName: String
    let circuitType: String
    let country: String?
    let totalLaps: Int?
}

struct SBPrediction: Codable {
    let raceId: String
    let driverId: String
    let driverName: String
    let teamId: String
    let predictedPosition: Double
    let probWinner: Double
    let probPodium: Double
    let probPoints: Double
    let probDnf: Double
    let expectedPoints: Double
    let simMedianPosition: Int?
    let simPosition25: Int?
    let simPosition75: Int?
    let confidence: String
}

struct SBStanding: Codable {
    let season: Int
    let driverId: String
    let driverName: String
    let teamId: String
    let position: Int
    let points: Double
    let wins: Int
    let championshipProb: Double?
}

struct SBConstructorStanding: Codable {
    let season: Int
    let teamId: String
    let teamName: String
    let position: Int
    let points: Double
    let championshipProb: Double?
}

struct SBModelMovement: Codable {
    let driverId: String
    let driverName: String
    let teamId: String
    let metric: String
    let delta: Double
    let reason: String
}

struct SBRaceAccuracy: Codable {
    let raceId: String
    let winnerPredicted: String
    let winnerActual: String
    let winnerCorrect: Bool
    let podiumCorrect: Int
    let top10Correct: Int
}

struct SBSeasonAccuracy: Codable {
    let season: Int
    let totalRaces: Int
    let winnerRate: Double
    let podiumRate: Double
    let top10Rate: Double
}

struct SBEloRating: Codable {
    let driverId: String
    let driverName: String
    let teamId: String
    let eloOverall: Double
    let eloQualifying: Double
    let eloCircuitType: Double
    let eloConstructor: Double
    let rank: Int
    let movementDelta: Int?
    let movementReason: String?
}

struct SBCircuit: Codable {
    let id: String
    let name: String
    let country: String?
    let circuitType: String
    let laps: Int?
    let lengthKm: Double?
    let raceDistanceKm: Double?
    let lapRecord: String?
    let gridCorrelation: Double?
    let overtakingRate: Double?
    let attritionRate: Double?
    let gridImportance: Double?
    let frontRowWinRate: Double?
    let description: String?
}

struct SBSessionSchedule: Codable {
    let session: String
    let day: String
    let time: String
}

struct SBWeather: Codable {
    let tempC: Int?
    let rainPct: Int?
    let windKmh: Int?
    let humidityPct: Int?
}

struct SBTrackSpecialist: Codable {
    let driverId: String
    let driverName: String
    let teamId: String
    let avgPosition: Double
    let races: Int
}

struct SBRecentWinner: Codable {
    let season: Int
    let driverName: String
    let teamId: String
}

struct SBNews: Codable {
    let source: String
    let title: String
    let impact: String?
}
