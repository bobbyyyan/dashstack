import SwiftUI

enum Theme {
    static let accent = Color(red: 0.93, green: 0.42, blue: 0.20)   // warm dashcam-orange
    static let accentSoft = Color(red: 0.95, green: 0.55, blue: 0.30).opacity(0.18)
    static let cardStroke = Color.primary.opacity(0.07)
    static let subtleText = Color.secondary.opacity(0.9)

    static let titleFont: Font = .system(.title, design: .rounded).weight(.semibold)
    static let sectionFont: Font = .system(.headline, design: .rounded)
    static let monoCaption: Font = .system(.caption, design: .monospaced)
}

/// Reusable rounded card container with material background.
struct Card<Content: View>: View {
    var padding: CGFloat = 18
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Theme.cardStroke, lineWidth: 1)
            )
    }
}

struct SectionLabel: View {
    let title: String
    let systemImage: String?

    init(_ title: String, systemImage: String? = nil) {
        self.title = title
        self.systemImage = systemImage
    }

    var body: some View {
        HStack(spacing: 6) {
            if let systemImage {
                Image(systemName: systemImage)
                    .foregroundStyle(Theme.accent)
            }
            Text(title)
                .font(Theme.sectionFont)
        }
    }
}

/// Pill that displays a key/value stat.
struct StatPill: View {
    let label: String
    let value: String
    var systemImage: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Theme.accent)
            }
            Text(value)
                .font(.system(.body, design: .rounded).weight(.semibold))
                .monospacedDigit()
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(Theme.accentSoft, in: Capsule())
    }
}

/// Primary call-to-action button.
struct PrimaryButtonStyle: ButtonStyle {
    var prominent: Bool = true
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.body, design: .rounded).weight(.semibold))
            .padding(.horizontal, 22)
            .padding(.vertical, 12)
            .background(
                LinearGradient(
                    colors: prominent
                        ? [Theme.accent, Theme.accent.opacity(0.82)]
                        : [Color.gray.opacity(0.25), Color.gray.opacity(0.18)],
                    startPoint: .top,
                    endPoint: .bottom
                ),
                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
            )
            .foregroundStyle(prominent ? Color.white : Color.primary)
            .shadow(color: prominent ? Theme.accent.opacity(0.35) : .clear,
                    radius: configuration.isPressed ? 2 : 8, y: configuration.isPressed ? 1 : 3)
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct GhostButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.body, design: .rounded).weight(.medium))
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Theme.cardStroke, lineWidth: 1)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(configuration.isPressed ? Color.primary.opacity(0.06) : .clear)
                    )
            )
    }
}
