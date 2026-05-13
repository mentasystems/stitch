// Package bench provides the prompt corpus and measurement harness for the
// fp16 vs int8 stitch comparison.
package bench

// Tools is the catalogue used for every prompt in the benchmark — chosen to
// mirror the oye personal-assistant demo.
//
// global-ok: read-only catalogue.
type ToolSpec struct {
	Name       string
	Parameters map[string]string
}

// BenchTools mirrors the oye handler catalogue (5 tools).
//
// global-ok: read-only catalogue shared across the benchmark.
var BenchTools = []ToolSpec{
	{Name: "get_weather", Parameters: map[string]string{"location": "string"}},
	{Name: "set_timer", Parameters: map[string]string{"seconds": "number", "label": "string"}},
	{Name: "play_music", Parameters: map[string]string{"query": "string"}},
	{Name: "add_to_list", Parameters: map[string]string{"list": "string", "item": "string"}},
	{Name: "open_app", Parameters: map[string]string{"name": "string"}},
}

// Prompts is a 60-prompt benchmark corpus chosen to exercise:
//   - the 5 tool families above
//   - varied phrasing (full questions, imperatives, ellipsis)
//   - both English and Spanish
//   - a small number of out-of-distribution queries that ideally should
//     produce [] (the model is allowed to fail gracefully on those).
//
// global-ok: immutable test corpus.
var Prompts = []string{
	// --- weather (12) ---
	"What's the weather in Madrid?",
	"weather in Tokyo",
	"weather in Lagos",
	"How's the weather in Paris",
	"Tell me the weather in Buenos Aires",
	"forecast for Berlin",
	"is it raining in London",
	"temperature in Tokyo right now",
	"¿Qué tiempo hace en Sevilla?",
	"¿Qué tiempo hace en Madrid?",
	"¿Hace frío en Barcelona?",
	"clima en Ciudad de México",

	// --- timer (10) ---
	"Set a 25 minute timer for pasta",
	"set timer 25 min",
	"Timer 30 minutes for pasta",
	"Start a 10 minute timer",
	"Remind me in 15 minutes",
	"set a 5 minute timer",
	"timer 1 hour",
	"alarma 10 minutos pasta",
	"pon un temporizador de 20 minutos",
	"recuérdame en 5 minutos",

	// --- music (10) ---
	"Play Despacito by Luis Fonsi",
	"Play Despacito",
	"Put on Bad Bunny",
	"play some jazz",
	"play Bohemian Rhapsody",
	"queue up Taylor Swift",
	"Pon a Bad Bunny",
	"pon a Rosalía",
	"reproduce Despacito",
	"music: lo-fi beats",

	// --- list (10) ---
	"add milk to groceries",
	"Add milk to groceries list",
	"Add bread to the shopping list",
	"add eggs to shopping",
	"put cheese on the groceries list",
	"add 'call dentist' to my todo",
	"añade leche a la lista de la compra",
	"agrega pan a la lista",
	"añade huevos a la compra",
	"add Madrid to bucket list",

	// --- open app (8) ---
	"Open Spotify",
	"Open Calculator",
	"launch Spotify",
	"open Calendar",
	"start Mail",
	"abre Spotify",
	"abre la calculadora",
	"open Notes",

	// --- out-of-distribution / hard (10) ---
	"What is the meaning of life?",
	"who won the world cup in 1998?",
	"translate hello to French",
	"summarize the news",
	"what's 47 times 13",
	"tell me a joke",
	"what time is it",
	"how do I cook risotto",
	"explica la teoría de la relatividad",
	"escríbeme un poema",
}
