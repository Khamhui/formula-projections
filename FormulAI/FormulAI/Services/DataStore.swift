import SwiftUI

@MainActor @Observable
final class DataStore {
    // Races
    var races: [RaceWeekend] = []

    // Core prediction data
    var predictions: [DriverPrediction] = []
    var driverStandings: [DriverStanding] = []
    var constructorStandings: [ConstructorStanding] = []
    var modelMovements: [ModelMovement] = []
    var raceAccuracy: [RaceAccuracy] = []
    var seasonAccuracy: SeasonAccuracy = SeasonAccuracy(winnerRate: 0, podiumRate: 0, top10Rate: 0, totalRaces: 0)

    // ELO ratings
    var eloRatings: [DriverELO] = []

    // Circuit data
    var circuitInfo: CircuitInfo?
    var circuitProfile: CircuitProfile?
    var sessionSchedule: [SessionScheduleEntry] = []
    var weather: WeatherForecast?
    var trackSpecialists: [TrackSpecialist] = []
    var recentWinners: [RecentWinner] = []

    // News
    var newsHeadlines: [NewsHeadline] = []

    // Derived data (cached, rebuilt on fetch)
    var predictionInsight: PredictionInsight = PredictionInsight(winnerId: "", whySentence: "Loading predictions...", casualDescription: "")
    var teammateH2H: [TeammateH2H] = []

    var whoToWatch: [WhoToWatch] {
        Array(predictions.prefix(3).map { p in
            let reason: String
            if let elo = eloRatings.first(where: { $0.id == p.id }) {
                reason = elo.movementReason.isEmpty ? "Top prediction for this race" : elo.movementReason
            } else {
                reason = "Strong predicted finish at P\(Int(p.predictedPosition))"
            }
            return WhoToWatch(name: p.driverName, teamId: p.teamId, insight: reason)
        })
    }

    var championshipProbabilities: [ChampionshipProbability] {
        driverStandings
            .filter { ($0.championshipProb ?? 0) > 0 }
            .sorted { ($0.championshipProb ?? 0) > ($1.championshipProb ?? 0) }
            .map { s in
                ChampionshipProbability(id: s.id, driverName: s.driverName, teamId: s.teamId, probability: s.championshipProb ?? 0)
            }
    }

    // State
    var isLoading = false
    var lastFetched: Date?
    var errors: [String] = []
    var error: String? { errors.first }

    func fetchAll() async {
        isLoading = true
        errors = []

        async let t1 = fetchRaces()
        async let t2 = fetchPredictions()
        async let t3 = fetchStandings()
        async let t4 = fetchConstructorStandings()
        async let t5 = fetchMovements()
        async let t6 = fetchEloRatings()
        async let t7 = fetchCircuitData()
        async let t8 = fetchNews()

        await t1; await t2; await t3; await t4; await t5; await t6; await t7; await t8

        rebuildDerivedData()
        isLoading = false
        lastFetched = Date()
    }

    private func rebuildDerivedData() {
        rebuildPredictionInsight()
        rebuildTeammateH2H()
    }

    private func rebuildPredictionInsight() {
        guard let winner = predictions.first else { return }
        predictionInsight = PredictionInsight(
            winnerId: winner.id,
            whySentence: "\(winner.teamName)'s technical circuit mastery and \(winner.driverName)'s qualifying edge give him a clear advantage at the upcoming race.",
            casualDescription: "About 1 in \(max(2, Int(round(100.0 / max(winner.simWinPct, 1))))) chance of winning"
        )
    }

    private func rebuildTeammateH2H() {
        let teams = Dictionary(grouping: driverStandings, by: \.teamId)
        teammateH2H = teams.compactMap { teamId, drivers in
            guard drivers.count >= 2 else { return nil }
            let sorted = drivers.sorted { $0.position < $1.position }
            let d1 = sorted[0], d2 = sorted[1]
            return TeammateH2H(
                driver1Id: d1.id, driver1Name: d1.driverName,
                driver2Id: d2.id, driver2Name: d2.driverName,
                teamId: teamId, teamName: F1Team.from(apiId: teamId)?.displayName ?? teamId.capitalized,
                driver1QualiWins: d1.wins, driver2QualiWins: d2.wins,
                driver1RaceWins: d1.wins, driver2RaceWins: d2.wins
            )
        }.sorted { $0.teamName < $1.teamName }
    }

    // MARK: - Fetchers

    private func fetchRaces() async {
        do {
            let results = try await SupabaseClient.fetch(
                "races", query: "season=eq.2026&order=round&select=*", as: [SBRace].self
            )
            if !results.isEmpty {
                races = results.map { r in
                    RaceWeekend(
                        id: r.id, season: r.season, round: r.round, name: r.name,
                        circuitType: CircuitType(rawValue: r.circuitType) ?? .mixed,
                        date: nil, hasPrediction: true,
                        circuitId: r.circuitId
                    )
                }
            }
        } catch { }
    }

    private func fetchPredictions() async {
        do {
            let results = try await SupabaseClient.fetch(
                "predictions", query: "order=predicted_position&select=*", as: [SBPrediction].self
            )
            let grouped = Dictionary(grouping: results, by: \.raceId)
            let latest = grouped[grouped.keys.sorted().last ?? ""] ?? []
            if !latest.isEmpty {
                predictions = latest.map { p in
                    DriverPrediction(
                        id: p.driverId, driverName: p.driverName, teamId: p.teamId,
                        teamName: teamDisplayName(p.teamId), grid: nil,
                        predictedPosition: p.predictedPosition,
                        simWinPct: p.probWinner * 100, simPodiumPct: p.probPodium * 100,
                        simPointsPct: p.probPoints * 100, simDnfPct: p.probDnf * 100,
                        simExpectedPoints: p.expectedPoints,
                        simMedianPosition: p.simMedianPosition ?? Int(p.predictedPosition),
                        simPosition25: p.simPosition25 ?? max(1, Int(p.predictedPosition) - 3),
                        simPosition75: p.simPosition75 ?? min(22, Int(p.predictedPosition) + 3),
                        probWinnerLo: nil, probWinnerHi: nil
                    )
                }
            }
        } catch { errors.append("Predictions: \(error.localizedDescription)") }
    }

    private func fetchStandings() async {
        do {
            let results = try await SupabaseClient.fetch(
                "standings", query: "season=eq.2026&order=position&select=*", as: [SBStanding].self
            )
            if !results.isEmpty {
                driverStandings = results.map { s in
                    DriverStanding(id: s.driverId, driverName: s.driverName, teamId: s.teamId,
                                   position: s.position, points: s.points, wins: s.wins,
                                   championshipProb: s.championshipProb)
                }
            }
        } catch { errors.append("Standings: \(error.localizedDescription)") }
    }

    private func fetchConstructorStandings() async {
        do {
            let results = try await SupabaseClient.fetch(
                "constructor_standings", query: "season=eq.2026&order=position&select=*", as: [SBConstructorStanding].self
            )
            if !results.isEmpty {
                constructorStandings = results.map { c in
                    ConstructorStanding(id: c.teamId, teamName: c.teamName, position: c.position,
                                        points: c.points, championshipProb: c.championshipProb)
                }
            }
        } catch { errors.append("Constructors: \(error.localizedDescription)") }
    }

    private func fetchMovements() async {
        do {
            let results = try await SupabaseClient.fetch(
                "model_movements", query: "order=created_at.desc&limit=10&select=*", as: [SBModelMovement].self
            )
            if !results.isEmpty {
                modelMovements = results.map { m in
                    ModelMovement(driverId: m.driverId, driverName: m.driverName, teamId: m.teamId,
                                  metric: MovementMetric(rawValue: m.metric) ?? .podium,
                                  delta: m.delta, reason: m.reason)
                }
            }
        } catch { }
    }

    private func fetchEloRatings() async {
        do {
            let results = try await SupabaseClient.fetch(
                "elo_ratings", query: "season=eq.2026&order=rank&select=*", as: [SBEloRating].self
            )
            if !results.isEmpty {
                eloRatings = results.map { e in
                    DriverELO(
                        id: e.driverId, driverName: e.driverName, teamId: e.teamId,
                        eloOverall: e.eloOverall, eloQualifying: e.eloQualifying,
                        eloCircuitType: e.eloCircuitType, eloConstructor: e.eloConstructor,
                        rank: e.rank,
                        history: generateEloHistory(base: e.eloOverall, delta: e.movementDelta ?? 0),
                        movementReason: e.movementReason ?? ""
                    )
                }
            }
        } catch { errors.append("ELO: \(error.localizedDescription)") }
    }

    private func fetchCircuitData() async {
        async let circuitTask: Void = fetchCircuit()
        async let scheduleTask: Void = fetchSessionSchedule()
        async let weatherTask: Void = fetchWeather()
        async let specialistsTask: Void = fetchTrackSpecialists()
        async let winnersTask: Void = fetchRecentWinners()

        await circuitTask; await scheduleTask; await weatherTask; await specialistsTask; await winnersTask

        if var info = circuitInfo, !recentWinners.isEmpty {
            info.recentWinners = recentWinners
            circuitInfo = info
        }
    }

    private func fetchCircuit() async {
        do {
            let circuits = try await SupabaseClient.fetch(
                "circuits", query: "id=eq.suzuka&select=*", as: [SBCircuit].self
            )
            if let c = circuits.first {
                circuitInfo = CircuitInfo(
                    name: c.name, country: c.country ?? "", laps: c.laps ?? 53,
                    lengthKm: c.lengthKm ?? 5.807, raceDistanceKm: c.raceDistanceKm ?? 307.5,
                    lapRecord: c.lapRecord ?? "—", description: c.description ?? "",
                    recentWinners: []
                )
                circuitProfile = CircuitProfile(
                    gridCorrelation: c.gridCorrelation ?? 0.5, overtakingRate: c.overtakingRate ?? 0.3,
                    attritionRate: c.attritionRate ?? 0.15, gridImportance: c.gridImportance ?? 0.7,
                    frontRowWinRate: c.frontRowWinRate ?? 0.65
                )
            }
        } catch { }
    }

    private func fetchSessionSchedule() async {
        do {
            let schedule = try await SupabaseClient.fetch(
                "session_schedule", query: "race_id=eq.2026-r03-suzuka&select=session,day,time", as: [SBSessionSchedule].self
            )
            if !schedule.isEmpty {
                sessionSchedule = schedule.map { SessionScheduleEntry(session: $0.session, day: $0.day, time: $0.time) }
            }
        } catch { }
    }

    private func fetchWeather() async {
        do {
            let results = try await SupabaseClient.fetch(
                "weather", query: "race_id=eq.2026-r03-suzuka&select=*", as: [SBWeather].self
            )
            if let w = results.first {
                weather = WeatherForecast(tempC: w.tempC ?? 0, rainPct: w.rainPct ?? 0,
                                          windKmh: w.windKmh ?? 0, humidityPct: w.humidityPct ?? 0)
            }
        } catch { }
    }

    private func fetchTrackSpecialists() async {
        do {
            let specs = try await SupabaseClient.fetch(
                "track_specialists", query: "circuit_id=eq.suzuka&order=avg_position&select=*", as: [SBTrackSpecialist].self
            )
            if !specs.isEmpty {
                trackSpecialists = specs.map { s in
                    TrackSpecialist(id: s.driverId, name: s.driverName, teamId: s.teamId,
                                    avgPos: s.avgPosition, races: s.races)
                }
            }
        } catch { }
    }

    private func fetchRecentWinners() async {
        do {
            let winners = try await SupabaseClient.fetch(
                "recent_winners", query: "circuit_id=eq.suzuka&order=season.desc&select=*", as: [SBRecentWinner].self
            )
            if !winners.isEmpty {
                recentWinners = winners.map { w in
                    RecentWinner(season: w.season, driver: w.driverName, teamId: w.teamId)
                }
            }
        } catch { }
    }

    private func fetchNews() async {
        do {
            let results = try await SupabaseClient.fetch(
                "news", query: "order=published_at.desc&limit=10&select=*", as: [SBNews].self
            )
            if !results.isEmpty {
                newsHeadlines = results.map { n in
                    NewsHeadline(source: n.source, title: n.title, timeAgo: "recent", impact: n.impact)
                }
            }
        } catch { }
    }

    // MARK: - Helpers

    private func teamDisplayName(_ teamId: String) -> String {
        F1Team.from(apiId: teamId)?.displayName ?? teamId.capitalized
    }

    private func generateEloHistory(base: Double, delta: Int) -> [Double] {
        (0..<6).map { i in base - Double(delta) * (1.0 - Double(i) / 5.0) + Double.random(in: -5...5) }
    }
}

// MARK: - Environment Key

private struct DataStoreKey: EnvironmentKey {
    static let defaultValue = DataStore()
}

extension EnvironmentValues {
    var dataStore: DataStore {
        get { self[DataStoreKey.self] }
        set { self[DataStoreKey.self] = newValue }
    }
}
