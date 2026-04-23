/**
 * Advanced Brand Review Sentiment Analyzer
 * Pure frontend NLP sentiment analysis application
 * Uses: Preprocessing, Tokenization, Stemming, Sentiment Scoring with Negation Handling
 * No backend required — all logic runs in the browser
 */

// ─── Brand List ───────────────────────────────────────────────────────────────
// All 8 brands whose reviews are loaded and analyzed by the application.
const BRANDS = [
  'Adidas',
  'Zara',
  'Dell',
  'Toyota Supra',
  'iPhone',
  'Lenskart',
  'Lloyd',
  'Titan',
];

// ─── Positive Sentiment Words ─────────────────────────────────────────────────
// Words that contribute a +1 score to a review's sentiment.
// When preceded by "not", the contribution is inverted to -1 (negation handling).
const POSITIVE_WORDS = new Set([
  'good',
  'great',
  'excellent',
  'amazing',
  'love',
  'best',
  'comfortable',
  'stylish',
]);

// ─── Negative Sentiment Words ─────────────────────────────────────────────────
// Words that contribute a -1 score to a review's sentiment.
// When preceded by "not", the contribution is inverted to +1 (negation handling).
// Note: 'worth' is included so the phrase "not worth" is handled via negation.
const NEGATIVE_WORDS = new Set([
  'bad',
  'worst',
  'poor',
  'hate',
  'expensive',
  'disappoint',
  'worth', // "not worth" → negation turns this into a negative signal
]);

// ─── Stop Words ───────────────────────────────────────────────────────────────
// Common English words that carry little semantic meaning and are removed
// during the preprocessing stage to reduce noise in sentiment analysis.
const STOPWORDS = new Set([
  'i', 'is', 'the', 'and', 'a', 'an', 'in', 'on', 'at',
  'to', 'for', 'of', 'with', 'by', 'from', 'as',
  'this', 'that', 'it', 'be', 'are', 'was', 'were',
  'been', 'have', 'has', 'had', 'do', 'does', 'did',
  'but', 'or', 'if', 'then', 'so', 'very', 'can',
  'will', 'just', 'should', 'now',
  'my', 'me', 'we', 'they', 'their', 'its', 'our', 'your',
  'he', 'she', 'his', 'her', 'them',
  'these', 'those', 'am', 'up', 'out',
  'about', 'after', 'all', 'also', 'any',
  'because', 'before', 'between', 'both',
  'each', 'few', 'more', 'most', 'other', 'over',
  'same', 'than', 'too', 'under', 'while',
  'who', 'whom', 'which', 'when', 'where', 'how', 'what',
  'there', 'here', 'no', 'only', 'own', 'such',
  'into', 'through', 'during', 'again', 'further', 'once',
]);

// ─── Application State ────────────────────────────────────────────────────────
// Centralized state object that drives all UI rendering.
// Mutated by event handlers; UI components read from this object.
const appState = {
  // Raw JSON data loaded from reviews.json — array of { brand, reviews[] } objects
  rawData: [],

  // Processed reviews keyed by brand name.
  // Map<string, ProcessedReview[]> where each ProcessedReview has:
  //   { originalText, tokens, score, label }
  processedReviews: new Map(),

  // The brand currently selected by the user (null before first selection)
  activeBrand: null,

  // The sentiment filter currently applied to the review list
  // One of: 'All' | 'Positive' | 'Negative' | 'Neutral'
  activeFilter: 'All',

  // Word frequency data keyed by brand name.
  // Map<string, Array<{ word: string, count: number }>>
  wordFrequencies: new Map(),
};

// ─── Data Loader ──────────────────────────────────────────────────────────────

/**
 * Loads and validates review data from reviews.json.
 *
 * @returns {Promise<Array<{brand: string, reviews: string[]}>>} Validated data array
 * @throws {Error} On network failure, invalid JSON, or wrong data structure
 */
async function loadReviewData() {
  // Step 1: Fetch reviews.json from the same directory as index.html
  let response;
  try {
    response = await fetch('reviews.json');
  } catch (networkError) {
    // Network-level failure (e.g., file not found, CORS, offline)
    throw new Error(
      'Failed to load reviews.json. Please ensure the file exists in the same directory as index.html.'
    );
  }

  // Step 2: Check HTTP status — a non-ok response (e.g., 404) is also a load failure
  if (!response.ok) {
    throw new Error(
      'Failed to load reviews.json. Please ensure the file exists in the same directory as index.html.'
    );
  }

  // Step 3: Parse the response body as JSON
  let data;
  try {
    data = await response.json();
  } catch (parseError) {
    // JSON.parse threw — the file contains malformed JSON
    throw new Error(
      'Invalid JSON format in reviews.json. Please check the file for syntax errors.'
    );
  }

  // Step 4: Validate the top-level structure is a non-empty array
  if (!Array.isArray(data)) {
    throw new Error(
      'Invalid data structure in reviews.json. Expected an array of {brand, reviews} objects.'
    );
  }

  // Step 5: Validate each element has the required { brand, reviews } shape
  const isValid = data.every(
    (item) =>
      item !== null &&
      typeof item === 'object' &&
      typeof item.brand === 'string' &&
      Array.isArray(item.reviews)
  );

  if (!isValid) {
    throw new Error(
      'Invalid data structure in reviews.json. Expected an array of {brand, reviews} objects.'
    );
  }

  // Step 6: Return the validated data array for further processing
  return data;
}

// ─── Error Display Helper ─────────────────────────────────────────────────────

/**
 * Displays a styled error message in the #error-container element and
 * clears all other main content areas to give the error full visibility.
 *
 * @param {string} message - The error message to display to the user
 */
function displayError(message) {
  // Locate the dedicated error container in the DOM
  const errorContainer = document.getElementById('error-container');

  // Clear any previously rendered error so we never stack multiple banners
  errorContainer.innerHTML = '';

  // Build the styled error block using Tailwind utility classes:
  //   bg-red-50      — light red background
  //   border          — visible border
  //   border-red-300  — red border colour
  //   text-red-700    — dark red text for readability
  //   rounded-lg      — rounded corners
  //   p-4             — comfortable padding
  //   flex / items-start — align icon and text side-by-side
  const errorBlock = document.createElement('div');
  errorBlock.className =
    'bg-red-50 border border-red-300 text-red-700 rounded-lg p-4 flex items-start gap-3';

  // Icon column — a simple SVG warning icon for visual emphasis
  const iconWrapper = document.createElement('div');
  iconWrapper.className = 'flex-shrink-0 mt-0.5';
  iconWrapper.innerHTML = `
    <svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"
         xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
    </svg>`;

  // Text column — bold heading plus the specific error message
  const textWrapper = document.createElement('div');

  const heading = document.createElement('p');
  heading.className = 'font-semibold text-red-800';
  heading.textContent = 'Error loading data';

  const body = document.createElement('p');
  body.className = 'mt-1 text-sm';
  // Use textContent (not innerHTML) to prevent XSS from untrusted message strings
  body.textContent = message;

  textWrapper.appendChild(heading);
  textWrapper.appendChild(body);

  errorBlock.appendChild(iconWrapper);
  errorBlock.appendChild(textWrapper);

  // Inject the error block into the container
  errorContainer.appendChild(errorBlock);

  // Clear all other content areas so the error is the only thing visible
  const reviewCards   = document.getElementById('review-cards');
  const dashboard     = document.getElementById('dashboard');
  const filterBar     = document.getElementById('filter-bar');
  const wordFrequency = document.getElementById('word-frequency');

  if (reviewCards)   reviewCards.innerHTML   = '';
  if (dashboard)     dashboard.innerHTML     = '';
  if (filterBar)     filterBar.innerHTML     = '';
  if (wordFrequency) wordFrequency.innerHTML = '';
}

// ─── NLP Engine ───────────────────────────────────────────────────────────────

/**
 * Preprocesses raw review text for NLP analysis.
 *
 * Steps:
 *   1. Lowercase the entire string so comparisons are case-insensitive.
 *   2. Strip every character that is not a letter (a-z) or a space, removing
 *      punctuation, digits, and special symbols.
 *   3. Split on whitespace, then filter out stopwords and empty strings so
 *      only meaningful content words remain.
 *   4. Rejoin the surviving tokens with a single space and return the result.
 *
 * @param {string} text       - Raw review text (may contain mixed case, punctuation, etc.)
 * @param {Set<string>} stopwords - Set of lowercase stopword strings to remove
 * @returns {string} Cleaned, lowercase, stopword-free text
 *
 * Requirements: 3.1, 3.2, 3.3, 3.4
 */
function preprocess(text, stopwords) {
  // Step 1: Convert to lowercase so all comparisons are case-insensitive
  let cleaned = text.toLowerCase();

  // Step 2: Remove every character that is not a lowercase letter or a space
  // This strips punctuation (!, ., ,), digits, and any other special characters
  cleaned = cleaned.replace(/[^a-z\s]/g, '');

  // Step 3: Split on whitespace into individual word tokens, then:
  //   - filter out empty strings produced by consecutive spaces
  //   - filter out any token that appears in the stopwords set
  const tokens = cleaned
    .split(/\s+/)
    .filter((token) => token.length > 0 && !stopwords.has(token));

  // Step 4: Rejoin the remaining meaningful tokens with a single space
  return tokens.join(' ');
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tokenizes preprocessed text into an array of word tokens.
 *
 * Steps:
 *   1. Split the input string on one or more whitespace characters.
 *   2. Filter out any empty strings that result from leading/trailing or
 *      consecutive whitespace in the input.
 *
 * @param {string} text - Preprocessed (clean, lowercase) text
 * @returns {string[]} Array of non-empty word tokens
 *
 * Requirements: 4.1, 4.2
 */
function tokenize(text) {
  // Step 1: Split on one or more whitespace characters (\s+) to handle
  // multiple consecutive spaces, tabs, or newlines gracefully
  const rawTokens = text.split(/\s+/);

  // Step 2: Filter out empty strings — these appear when the input has
  // leading/trailing whitespace or when split produces an empty first/last element
  return rawTokens.filter((token) => token.length > 0);
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reduces a token to its approximate root form using suffix-stripping rules
 * (Porter-like stemming).
 *
 * Rules are applied in priority order; only the FIRST matching rule fires.
 * A rule only applies when the remaining stem would be longer than 2 characters.
 *
 * Priority order:
 *   1. '-ing'  → remove 'ing'  (e.g. "disappointing" → "disappoint")
 *   2. '-ed'   → remove 'ed'   (e.g. "disappointed"  → "disappoint")
 *   3. '-ly'   → remove 'ly'   (e.g. "quickly"       → "quick")
 *   4. '-er'   → remove 'er'   (e.g. "faster"        → "fast")
 *   5. '-est'  → remove 'est'  (e.g. "fastest"       → "fast")
 *   6. '-s'    → remove 's'    (e.g. "loves"         → "love")
 *              but NOT '-ss'   (e.g. "lass" stays "lass")
 *
 * Special 'e'-restoration: after stripping '-ing' or '-ed', if the resulting
 * stem ends in a single consonant preceded by a vowel (CVC pattern at the end),
 * restore the silent 'e' (e.g. "lov" → "love", "us" is too short so unchanged).
 *
 * @param {string} token - A single lowercase word token
 * @returns {string} The stemmed token (never longer than the original)
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4
 */
function stem(token) {
  // Helper: returns true if the character is a vowel (a, e, i, o, u)
  const isVowel = (ch) => 'aeiou'.includes(ch);

  // Helper: after stripping '-ing' or '-ed', check whether the stem ends in
  // a vowel-consonant pattern and restore the silent 'e' if so.
  // Example: "lov" ends in vowel('o') + consonant('v') → restore → "love"
  // Example: "disappoint" ends in consonant('n') + consonant('t') → no restore
  const restoreE = (stem) => {
    const len = stem.length;
    if (len >= 2 && !isVowel(stem[len - 1]) && isVowel(stem[len - 2])) {
      return stem + 'e';
    }
    return stem;
  };

  // Rule 1: Strip '-ing' if the remaining stem is longer than 2 characters
  // e.g. "loving" (6) → strip 3 → "lov" (3 > 2) ✓ → restore 'e' → "love"
  // e.g. "disappointing" → "disappoint" (10 > 2) ✓ → no 'e' restore needed
  if (token.endsWith('ing') && token.length - 3 > 2) {
    return restoreE(token.slice(0, -3));
  }

  // Rule 2: Strip '-ed' if the remaining stem is longer than 2 characters
  // e.g. "loved" (5) → strip 2 → "lov" (3 > 2) ✓ → restore 'e' → "love"
  // e.g. "disappointed" → "disappoint" (10 > 2) ✓ → no 'e' restore needed
  if (token.endsWith('ed') && token.length - 2 > 2) {
    return restoreE(token.slice(0, -2));
  }

  // Rule 3: Strip '-ly' if the remaining stem is longer than 2 characters
  // e.g. "quickly" → "quick"
  if (token.endsWith('ly') && token.length - 2 > 2) {
    return token.slice(0, -2);
  }

  // Rule 4: Strip '-er' if the remaining stem is longer than 2 characters
  // e.g. "faster" → "fast"
  if (token.endsWith('er') && token.length - 2 > 2) {
    return token.slice(0, -2);
  }

  // Rule 5: Strip '-est' if the remaining stem is longer than 2 characters
  // e.g. "fastest" → "fast"
  if (token.endsWith('est') && token.length - 3 > 2) {
    return token.slice(0, -3);
  }

  // Rule 6: Strip '-s' (plurals) if the remaining stem is longer than 2 characters
  // but NOT when the word ends in '-ss' (e.g. "lass", "class" should not be stripped)
  // e.g. "loves" → "love", "cats" → "cat"
  // e.g. "lass" → "lass" (ends in 'ss', skip)
  if (token.endsWith('s') && !token.endsWith('ss') && token.length - 1 > 2) {
    return token.slice(0, -1);
  }

  // No rule matched — return the token unchanged
  return token;
}

// ─────────────────────────────────────────────────────────────────────────────

/**
 * Scores the sentiment of a token array and returns a numeric score and label.
 *
 * Algorithm:
 *   - Start with score = 0.
 *   - For each token, check membership in positiveWords and negativeWords.
 *   - Negation handling: if the immediately preceding token (index i-1) is "not",
 *     the sentiment contribution is INVERTED:
 *       • positive word after "not" → score -= 1  (e.g. "not good" is negative)
 *       • negative word after "not" → score += 1  (e.g. "not bad"  is positive)
 *   - Without negation:
 *       • positive word → score += 1
 *       • negative word → score -= 1
 *   - Label is determined from the final score:
 *       • score > 0  → 'Positive'
 *       • score < 0  → 'Negative'
 *       • score === 0 → 'Neutral'
 *
 * @param {string[]} tokens           - Array of stemmed, preprocessed tokens
 * @param {Set<string>} positiveWords - Set of positive sentiment words
 * @param {Set<string>} negativeWords - Set of negative sentiment words
 * @returns {{ score: number, label: string }} Sentiment score and label
 *
 * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2, 7.3
 */
function scoreSentiment(tokens, positiveWords, negativeWords) {
  // Initialize the running sentiment score to neutral (zero)
  let score = 0;

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];

    // Look up the token immediately before the current one to detect negation.
    // When i === 0 there is no previous token, so prevToken is undefined.
    const prevToken = i > 0 ? tokens[i - 1] : undefined;

    // Check if the current token is a positive sentiment word
    if (positiveWords.has(token)) {
      if (prevToken === 'not') {
        // Negation detected: "not <positive>" → treat as negative contribution
        score -= 1;
      } else {
        // Normal positive word → add to score
        score += 1;
      }
    }

    // Check if the current token is a negative sentiment word
    // (A token could theoretically be in both sets, so we use separate if blocks)
    if (negativeWords.has(token)) {
      if (prevToken === 'not') {
        // Negation detected: "not <negative>" → treat as positive contribution
        score += 1;
      } else {
        // Normal negative word → subtract from score
        score -= 1;
      }
    }
  }

  // Determine the sentiment label based on the sign of the final score
  let label;
  if (score > 0) {
    label = 'Positive';
  } else if (score < 0) {
    label = 'Negative';
  } else {
    // score === 0: no net sentiment detected
    label = 'Neutral';
  }

  return { score, label };
}

// ─── Review Processing Pipeline ───────────────────────────────────────────────

/**
 * Runs the full NLP pipeline over every review in rawData and returns a Map
 * of processed results keyed by brand name.
 *
 * Pipeline per review text:
 *   1. preprocess(text, STOPWORDS)  — lowercase, strip punctuation, remove stopwords
 *   2. tokenize(preprocessedText)   — split into word tokens
 *   3. tokens.map(stem)             — reduce each token to its root form
 *   4. scoreSentiment(stemmedTokens, POSITIVE_WORDS, NEGATIVE_WORDS)
 *                                   — compute numeric score and label
 *
 * Each result is stored as:
 *   { originalText, tokens, score, label }
 *
 * Defensive handling:
 *   - If a review entry is null or undefined, the review is skipped and a
 *     console.warn is emitted so the issue is visible without crashing the app.
 *
 * @param {Array<{brand: string, reviews: string[]}>} rawData
 *   Validated data array returned by loadReviewData()
 * @returns {Map<string, Array<{originalText: string, tokens: string[], score: number, label: string}>>}
 *   Map from brand name → array of processed review objects
 *
 * Requirements: 1.2, 11.1, 11.2, 11.4, 13.3
 */
function processAllReviews(rawData) {
  // Initialize the result Map; each key will be a brand name string
  const result = new Map();

  // Iterate over every brand entry in the raw data array
  for (const brandEntry of rawData) {
    const { brand, reviews } = brandEntry;

    // Accumulate processed review objects for this brand
    const processedForBrand = [];

    // Iterate over each review text belonging to this brand
    for (const reviewText of reviews) {
      // Defensive check: skip null or undefined review text rather than crashing
      if (reviewText == null) {
        console.warn(
          `[processAllReviews] Skipping null/undefined review for brand "${brand}".`
        );
        continue;
      }

      // Step 1: Preprocess — lowercase, remove punctuation, strip stopwords
      const preprocessedText = preprocess(reviewText, STOPWORDS);

      // Step 2: Tokenize — split the cleaned text into an array of word tokens
      const tokens = tokenize(preprocessedText);

      // Step 3: Stem — reduce each token to its approximate root form
      const stemmedTokens = tokens.map(stem);

      // Step 4: Score — compute sentiment score and label from stemmed tokens
      const { score, label } = scoreSentiment(stemmedTokens, POSITIVE_WORDS, NEGATIVE_WORDS);

      // Store the result object; originalText preserves the unmodified review
      processedForBrand.push({
        originalText: reviewText,
        tokens: stemmedTokens,
        score,
        label,
      });
    }

    // Map this brand's processed reviews into the result Map
    result.set(brand, processedForBrand);
  }

  // Return the fully populated Map for use by the application state
  return result;
}

// ─── Word Frequency Analyzer ──────────────────────────────────────────────────

/**
 * Computes word frequency statistics across an array of processed review objects
 * and returns the top 10 most frequent tokens in descending order.
 *
 * Algorithm:
 *   1. Walk every review in the array and every token in each review.
 *   2. Maintain a plain-object frequency map: { [word]: count }.
 *   3. Convert the map to an array of { word, count } objects.
 *   4. Sort the array by count descending (ties are left in insertion order).
 *   5. Return the first 10 entries (or fewer if there are fewer unique tokens).
 *
 * @param {Array<{originalText: string, tokens: string[], score: number, label: string}>} processedReviews
 *   Array of processed review objects for a single brand (not the full Map)
 * @returns {Array<{word: string, count: number}>}
 *   Up to 10 entries sorted by descending frequency
 *
 * Requirements: 11.1, 11.2, 11.4
 */
function computeWordFrequencies(processedReviews) {
  // Initialize an empty frequency map: word → occurrence count
  const freqMap = {};

  // Iterate over every processed review in the array
  for (const review of processedReviews) {
    // Iterate over every stemmed token in this review
    for (const token of review.tokens) {
      // Increment the count for this token, defaulting to 0 if unseen
      freqMap[token] = (freqMap[token] || 0) + 1;
    }
  }

  // Convert the frequency map to an array of { word, count } objects
  const freqArray = Object.entries(freqMap).map(([word, count]) => ({ word, count }));

  // Sort the array by count in descending order so the most frequent words come first
  freqArray.sort((a, b) => b.count - a.count);

  // Return only the top 10 entries (slice is safe even if array has fewer elements)
  return freqArray.slice(0, 10);
}

// ─── UI Renderer — Brand Selector ─────────────────────────────────────────────

/**
 * Renders one button per brand into the #brand-selector container.
 *
 * Active brand button receives a filled blue style with a subtle scale transform
 * to give clear visual feedback about the current selection.
 * Inactive brand buttons use a white/bordered style with a hover highlight.
 * Each button is wired to call selectBrand(brandName) on click.
 *
 * @param {string[]} brands      - Ordered list of brand name strings
 * @param {string}   activeBrand - The brand name that is currently selected
 * @returns {void}
 *
 * Requirements: 2.1, 2.4, 2.5, 12.1, 12.3
 */
function renderBrandSelector(brands, activeBrand) {
  // Locate the flex-wrap container that holds the brand buttons
  const container = document.getElementById('brand-selector');

  // Clear any previously rendered buttons before re-rendering
  container.innerHTML = '';

  // Tailwind classes shared by every brand button regardless of active state
  const baseClasses =
    'px-4 py-2 rounded-lg font-medium text-sm transition duration-200 cursor-pointer';

  // Tailwind classes applied only to the currently active brand button
  const activeClasses =
    'bg-blue-600 text-white shadow-md scale-105 transform';

  // Tailwind classes applied to every inactive brand button
  const inactiveClasses =
    'bg-white text-gray-700 border border-gray-200 hover:bg-blue-50 hover:border-blue-300';

  // Create one button element per brand and append it to the container
  for (const brandName of brands) {
    const btn = document.createElement('button');

    // Determine whether this button represents the active brand
    const isActive = brandName === activeBrand;

    // Combine base classes with the appropriate active/inactive classes
    btn.className = `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`;

    // Display the brand name as the button label
    btn.textContent = brandName;

    // Wire the click handler — calls the global selectBrand event handler
    btn.addEventListener('click', () => selectBrand(brandName));

    container.appendChild(btn);
  }
}

// ─── UI Renderer — Dashboard ──────────────────────────────────────────────────

/**
 * Computes aggregate sentiment counts from an array of processed reviews.
 *
 * @param {Array<{originalText: string, tokens: string[], score: number, label: string}>} reviews
 *   Processed review objects for the active brand
 * @returns {{ total: number, positive: number, negative: number, neutral: number }}
 *   Object containing the four aggregate counts
 *
 * Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
 */
function computeDashboardStats(reviews) {
  // Total is simply the length of the reviews array
  const total = reviews.length;

  // Count each sentiment label by filtering the array
  const positive = reviews.filter((r) => r.label === 'Positive').length;
  const negative = reviews.filter((r) => r.label === 'Negative').length;
  const neutral  = reviews.filter((r) => r.label === 'Neutral').length;

  return { total, positive, negative, neutral };
}

/**
 * Renders four stat cards (Total, Positive, Negative, Neutral) into the
 * #dashboard container using the counts returned by computeDashboardStats.
 *
 * Each card uses a distinct color theme:
 *   Total    — blue  (bg-blue-50  / border-blue-200  / text-blue-700)
 *   Positive — green (bg-green-50 / border-green-200 / text-green-700)
 *   Negative — red   (bg-red-50   / border-red-200   / text-red-700)
 *   Neutral  — gray  (bg-gray-50  / border-gray-200  / text-gray-600)
 *
 * @param {Array<{originalText: string, tokens: string[], score: number, label: string}>} reviews
 *   Processed review objects for the active brand
 * @returns {void}
 *
 * Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 12.1
 */
function renderDashboard(reviews) {
  // Locate the grid container for the stat cards
  const container = document.getElementById('dashboard');

  // Clear any previously rendered cards
  container.innerHTML = '';

  // Compute the four aggregate counts from the reviews array
  const { total, positive, negative, neutral } = computeDashboardStats(reviews);

  // Card definitions: each entry describes one stat card's content and theme
  const cards = [
    {
      label: 'Total Reviews',
      count: total,
      icon: '📋',
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-700',
    },
    {
      label: 'Positive',
      count: positive,
      icon: '😊',
      bg: 'bg-green-50',
      border: 'border-green-200',
      text: 'text-green-700',
    },
    {
      label: 'Negative',
      count: negative,
      icon: '😞',
      bg: 'bg-red-50',
      border: 'border-red-200',
      text: 'text-red-700',
    },
    {
      label: 'Neutral',
      count: neutral,
      icon: '😐',
      bg: 'bg-gray-50',
      border: 'border-gray-200',
      text: 'text-gray-600',
    },
  ];

  // Build and append one card element per stat definition
  for (const card of cards) {
    // Outer card wrapper — colored background, border, rounded corners, padding
    const cardEl = document.createElement('div');
    cardEl.className = `${card.bg} ${card.border} border rounded-xl p-4 flex flex-col items-center justify-center text-center shadow-sm`;

    // Small icon/emoji at the top of the card
    const iconEl = document.createElement('div');
    iconEl.className = 'text-2xl mb-1';
    iconEl.textContent = card.icon;

    // Large count number — the primary data point on the card
    const countEl = document.createElement('div');
    countEl.className = `text-3xl font-bold ${card.text}`;
    countEl.textContent = card.count;

    // Label below the count describing what the number represents
    const labelEl = document.createElement('div');
    labelEl.className = `text-xs font-semibold uppercase tracking-wide ${card.text} mt-1 opacity-80`;
    labelEl.textContent = card.label;

    cardEl.appendChild(iconEl);
    cardEl.appendChild(countEl);
    cardEl.appendChild(labelEl);

    container.appendChild(cardEl);
  }
}

// ─── UI Renderer — Filter Bar ─────────────────────────────────────────────────

/**
 * Renders four sentiment filter buttons ("All", "Positive", "Negative", "Neutral")
 * into the #filter-bar container.
 *
 * The active filter button receives a filled color matching its sentiment:
 *   All      → bg-blue-600  text-white
 *   Positive → bg-green-600 text-white
 *   Negative → bg-red-600   text-white
 *   Neutral  → bg-gray-500  text-white
 *
 * Inactive buttons share a neutral white/bordered style.
 * Each button calls setFilter(filterName) on click.
 *
 * @param {string} activeFilter - The currently active filter value
 * @returns {void}
 *
 * Requirements: 10.1, 10.6, 12.1
 */
function renderFilterBar(activeFilter) {
  // Locate the flex-wrap container for the filter buttons
  const container = document.getElementById('filter-bar');

  // Clear any previously rendered filter buttons
  container.innerHTML = '';

  // Tailwind classes shared by every filter button
  const baseClasses =
    'px-4 py-2 rounded-lg text-sm font-medium transition duration-200';

  // Map each filter name to its active-state Tailwind color classes
  const activeColorMap = {
    All:      'bg-blue-600 text-white',
    Positive: 'bg-green-600 text-white',
    Negative: 'bg-red-600 text-white',
    Neutral:  'bg-gray-500 text-white',
  };

  // Tailwind classes for every inactive filter button
  const inactiveClasses =
    'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50';

  // The four filter options in display order
  const filters = ['All', 'Positive', 'Negative', 'Neutral'];

  for (const filterName of filters) {
    const btn = document.createElement('button');

    // Determine whether this button represents the currently active filter
    const isActive = filterName === activeFilter;

    // Apply active color or inactive style depending on selection state
    btn.className = `${baseClasses} ${isActive ? activeColorMap[filterName] : inactiveClasses}`;

    btn.textContent = filterName;

    // Wire the click handler — calls the global setFilter event handler
    btn.addEventListener('click', () => setFilter(filterName));

    container.appendChild(btn);
  }
}

// ─── UI Renderer — Review Cards ───────────────────────────────────────────────

/**
 * Filters a reviews array based on the active sentiment filter.
 *
 * When activeFilter is 'All', the full array is returned unchanged.
 * Otherwise only reviews whose label matches the filter are returned.
 *
 * @param {Array<{originalText: string, tokens: string[], score: number, label: string}>} reviews
 *   Full array of processed reviews for the active brand
 * @param {string} activeFilter - One of 'All' | 'Positive' | 'Negative' | 'Neutral'
 * @returns {Array<{originalText: string, tokens: string[], score: number, label: string}>}
 *   Filtered subset (or the original array when filter is 'All')
 *
 * Requirements: 10.2, 10.3, 10.4, 10.5
 */
function applyFilter(reviews, activeFilter) {
  // 'All' means no filtering — return the complete reviews array
  if (activeFilter === 'All') {
    return reviews;
  }

  // Otherwise keep only reviews whose label matches the selected filter
  return reviews.filter((review) => review.label === activeFilter);
}

/**
 * Renders review cards for the active brand into the #review-cards grid.
 *
 * Steps:
 *   1. Apply the active filter via applyFilter() to get the visible subset.
 *   2. If the filtered list is empty, show a "no results" message.
 *   3. Otherwise render one card per review containing:
 *        - Original review text
 *        - Color-coded sentiment badge (green / red / gray)
 *        - Numeric score with matching color
 *
 * @param {Array<{originalText: string, tokens: string[], score: number, label: string}>} reviews
 *   Full array of processed reviews for the active brand
 * @param {string} activeFilter - One of 'All' | 'Positive' | 'Negative' | 'Neutral'
 * @returns {void}
 *
 * Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 10.2, 10.3, 10.4, 10.5, 12.1, 12.5
 */
function renderReviewCards(reviews, activeFilter) {
  // Locate the grid container for review cards
  const container = document.getElementById('review-cards');

  // Clear any previously rendered cards
  container.innerHTML = '';

  // Apply the active filter to get only the reviews that should be displayed
  const filtered = applyFilter(reviews, activeFilter);

  // Handle the empty state — no reviews match the current filter
  if (filtered.length === 0) {
    const emptyMsg = document.createElement('p');
    emptyMsg.className =
      'col-span-full text-center text-gray-500 text-sm py-8';
    emptyMsg.textContent = 'No reviews match the selected filter.';
    container.appendChild(emptyMsg);
    return;
  }

  // Sentiment-to-color mapping for badge and score display
  // Each entry provides the badge background/text classes and the score text color
  const sentimentStyles = {
    Positive: {
      badge: 'bg-green-100 text-green-800',
      score: 'text-green-600',
    },
    Negative: {
      badge: 'bg-red-100 text-red-800',
      score: 'text-red-600',
    },
    Neutral: {
      badge: 'bg-gray-100 text-gray-600',
      score: 'text-gray-500',
    },
  };

  // Render one card element per filtered review
  for (const review of filtered) {
    // Outer card wrapper — white background, rounded corners, shadow, border, padding
    const card = document.createElement('div');
    card.className =
      'bg-white rounded-xl shadow-sm border border-gray-100 p-4 flex flex-col gap-3 transition duration-200';

    // Review text paragraph — main content of the card
    const textEl = document.createElement('p');
    textEl.className = 'text-gray-700 text-sm leading-relaxed flex-1';
    // Use textContent to prevent XSS from review text
    textEl.textContent = review.originalText;

    // Footer row — holds the sentiment badge and score side by side
    const footer = document.createElement('div');
    footer.className = 'flex items-center justify-between gap-2 mt-auto';

    // Sentiment badge — color-coded pill showing the label
    const styles = sentimentStyles[review.label] || sentimentStyles.Neutral;

    const badge = document.createElement('span');
    badge.className = `inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${styles.badge}`;
    badge.textContent = review.label;

    // Numeric score display — prefixed with + for positive scores
    const scoreEl = document.createElement('span');
    scoreEl.className = `text-xs font-medium ${styles.score}`;
    const scorePrefix = review.score > 0 ? '+' : '';
    scoreEl.textContent = `Score: ${scorePrefix}${review.score}`;

    footer.appendChild(badge);
    footer.appendChild(scoreEl);

    card.appendChild(textEl);
    card.appendChild(footer);

    container.appendChild(card);
  }
}

// ─── UI Renderer — Word Frequency Panel ───────────────────────────────────────

/**
 * Renders the top word frequency entries into the #word-frequency container.
 *
 * Each entry is displayed as a row containing:
 *   - The word label (left-aligned)
 *   - A count badge (right-aligned)
 *   - A proportional horizontal bar whose width is (count / maxCount) * 100%
 *
 * When the frequencies array is empty, a "no data" message is shown instead.
 *
 * @param {Array<{word: string, count: number}>} frequencies
 *   Up to 10 entries sorted by descending frequency (from computeWordFrequencies)
 * @returns {void}
 *
 * Requirements: 11.1, 11.2, 11.3, 11.4, 12.1
 */
function renderWordFrequency(frequencies) {
  // Locate the word frequency panel container
  const container = document.getElementById('word-frequency');

  // Clear any previously rendered rows
  container.innerHTML = '';

  // Handle the empty state — no frequency data available for this brand
  if (!frequencies || frequencies.length === 0) {
    const emptyMsg = document.createElement('p');
    emptyMsg.className = 'text-gray-400 text-sm text-center py-4';
    emptyMsg.textContent = 'No word frequency data available.';
    container.appendChild(emptyMsg);
    return;
  }

  // Determine the highest count so bar widths can be computed proportionally
  const maxCount = frequencies[0].count;

  // Render one row per frequency entry
  for (const { word, count } of frequencies) {
    // Outer row wrapper — flex layout with label and count on the same line
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between mb-2';

    // Left side: word label
    const wordLabel = document.createElement('span');
    wordLabel.className = 'text-sm text-gray-700 font-medium w-24 truncate';
    wordLabel.textContent = word;

    // Right side: count badge
    const countBadge = document.createElement('span');
    countBadge.className =
      'text-xs font-semibold text-blue-600 bg-blue-50 rounded-full px-2 py-0.5 ml-2 shrink-0';
    countBadge.textContent = count;

    // Bar container — takes up the remaining horizontal space between label and badge
    const barContainer = document.createElement('div');
    barContainer.className = 'flex-1 mx-2 bg-gray-100 rounded-full h-2 overflow-hidden';

    // Proportional fill bar — width is a percentage of the maximum count
    const barFill = document.createElement('div');
    const widthPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
    barFill.className = 'bg-blue-400 h-2 rounded-full';
    barFill.style.width = `${widthPercent}%`;

    barContainer.appendChild(barFill);

    row.appendChild(wordLabel);
    row.appendChild(barContainer);
    row.appendChild(countBadge);

    container.appendChild(row);
  }
}

// ─── Event Handler — Brand Selection ─────────────────────────────────────────

/**
 * Handles a brand selection event: updates application state, computes word
 * frequencies for the newly selected brand, and re-renders all five UI
 * components to reflect the new selection.
 *
 * Steps:
 *   1. Set the active brand in state.
 *   2. Reset the active filter to 'All' so the new brand shows all reviews.
 *   3. Retrieve the processed reviews for the selected brand (or empty array).
 *   4. Compute and cache word frequencies for the brand.
 *   5. Re-render all five UI components.
 *   6. Clear any previously displayed error message.
 *
 * @param {string} brandName - The brand name that was clicked / selected
 * @returns {void}
 *
 * Requirements: 2.3, 2.4, 9.5, 10.7, 11.3, 12.4
 */
function selectBrand(brandName) {
  // Step 1: Update the active brand in centralized state
  appState.activeBrand = brandName;

  // Step 2: Reset the filter to 'All' whenever the brand changes (Req 10.7)
  appState.activeFilter = 'All';

  // Step 3: Retrieve the processed reviews for this brand; default to empty array
  // if the brand has no reviews (e.g., newly uploaded brand with no data yet)
  const reviews = appState.processedReviews.get(brandName) || [];

  // Step 4: Compute word frequencies for this brand and cache them in state
  // so subsequent renders don't need to recompute (Req 11.3)
  appState.wordFrequencies.set(brandName, computeWordFrequencies(reviews));

  // Step 5a: Re-render the brand selector so the active button is highlighted.
  // Use the full dynamic brand list from processedReviews so any uploaded brands
  // are also shown (not just the original static BRANDS array).
  const allBrands = appState.processedReviews.size > 0
    ? [...appState.processedReviews.keys()]
    : BRANDS;
  renderBrandSelector(allBrands, appState.activeBrand);

  // Step 5b: Re-render the dashboard stat cards with the new brand's counts
  renderDashboard(reviews);

  // Step 5c: Re-render the filter bar (reset to 'All' active state)
  renderFilterBar(appState.activeFilter);

  // Step 5d: Re-render the review cards grid (unfiltered for the new brand)
  renderReviewCards(reviews, appState.activeFilter);

  // Step 5e: Re-render the word frequency panel with the cached frequencies
  renderWordFrequency(appState.wordFrequencies.get(brandName));

  // Step 6: Clear any error banner that may have been shown previously
  document.getElementById('error-container').innerHTML = '';
}

// ─── Event Handler — Filter Selection ────────────────────────────────────────

/**
 * Handles a sentiment filter selection event: updates the active filter in
 * state and re-renders only the two UI components that depend on the filter
 * (the filter bar and the review cards grid).
 *
 * Steps:
 *   1. Update the active filter in state.
 *   2. Retrieve the processed reviews for the currently active brand.
 *   3. Re-render the filter bar so the correct button appears active.
 *   4. Re-render the review cards grid with the new filter applied.
 *
 * @param {string} filterName - One of 'All' | 'Positive' | 'Negative' | 'Neutral'
 * @returns {void}
 *
 * Requirements: 10.2, 10.3, 10.4, 10.5, 10.6
 */
function setFilter(filterName) {
  // Step 1: Persist the new filter value in centralized state
  appState.activeFilter = filterName;

  // Step 2: Get the current brand's reviews; default to empty array if none
  const reviews = appState.processedReviews.get(appState.activeBrand) || [];

  // Step 3: Re-render the filter bar so the newly selected button is highlighted
  renderFilterBar(appState.activeFilter);

  // Step 4: Re-render the review cards grid applying the new filter
  renderReviewCards(reviews, appState.activeFilter);
}

// ─── Toast Notification Helper ────────────────────────────────────────────────

/**
 * Displays a brief success toast notification in the #toast container.
 *
 * The toast auto-removes itself after 3 seconds with a CSS opacity fade-out.
 *
 * @param {string} message - The message text to display in the toast
 * @returns {void}
 */
function showToast(message) {
  // Locate the fixed toast container in the bottom-right corner of the viewport
  const toastContainer = document.getElementById('toast');

  // Create the toast element with Tailwind utility classes for styling
  const toast = document.createElement('div');
  toast.className =
    'bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg text-sm font-medium pointer-events-auto transition-opacity duration-500';
  // Use textContent to prevent XSS from the message string
  toast.textContent = message;

  // Append the toast to the container so it becomes visible
  toastContainer.appendChild(toast);

  // After 2.5 seconds begin the fade-out by setting opacity to 0
  setTimeout(() => {
    toast.style.opacity = '0';
  }, 2500);

  // After 3 seconds (fade complete) remove the element from the DOM
  setTimeout(() => {
    if (toast.parentNode) {
      toast.parentNode.removeChild(toast);
    }
  }, 3000);
}

// ─── File Upload Parser ───────────────────────────────────────────────────────

/**
 * Parses an uploaded file (JSON, CSV, or TXT) into the standard
 * `[{ brand, reviews }]` data format used by the application.
 *
 * Supported formats:
 *   - .json : Array of `{ brand: string, reviews: string[] }` objects
 *   - .csv  : Rows where column 0 = brand name, column 1 = review text.
 *             The header row is skipped if the first cell equals "brand".
 *             Rows are grouped by brand name.
 *   - .txt  : Lines in `"BrandName: review text"` format.
 *             Lines are grouped by brand name.
 *
 * Validation:
 *   - At least one brand with at least one review must be present.
 *   - Unsupported file extensions throw a descriptive error.
 *   - Empty or malformed content throws a descriptive error.
 *
 * @param {File} file - The File object selected by the user
 * @returns {Promise<Array<{brand: string, reviews: string[]}>>} Parsed data array
 * @throws {Error} On unsupported format, parse failure, or empty content
 */
function parseUploadedFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (event) => {
      const content = event.target.result;
      const fileName = file.name.toLowerCase();

      try {
        let parsedData;

        if (fileName.endsWith('.json')) {
          // ── JSON format ──────────────────────────────────────────────────
          // Expected: [{ brand: string, reviews: string[] }, ...]
          let jsonData;
          try {
            jsonData = JSON.parse(content);
          } catch {
            throw new Error(
              'Invalid JSON in uploaded file. Please check the file for syntax errors.'
            );
          }

          // Validate top-level structure is a non-empty array
          if (!Array.isArray(jsonData)) {
            throw new Error(
              'Invalid JSON format. Expected an array of { brand, reviews } objects.'
            );
          }

          // Validate each entry has the required shape
          const isValid = jsonData.every(
            (item) =>
              item !== null &&
              typeof item === 'object' &&
              typeof item.brand === 'string' &&
              Array.isArray(item.reviews)
          );

          if (!isValid) {
            throw new Error(
              'Invalid JSON format. Each entry must have a "brand" string and a "reviews" array.'
            );
          }

          parsedData = jsonData;

        } else if (fileName.endsWith('.csv')) {
          // ── CSV format ───────────────────────────────────────────────────
          // Column 0 = brand name, Column 1 = review text
          // Skip header row if first cell (lowercased) equals "brand"
          const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);

          // Determine whether the first row is a header
          let startIndex = 0;
          if (lines.length > 0) {
            const firstCell = lines[0].split(',')[0].trim().toLowerCase();
            if (firstCell === 'brand') {
              startIndex = 1; // skip header row
            }
          }

          // Group rows by brand name into a Map
          const brandMap = new Map();
          for (let i = startIndex; i < lines.length; i++) {
            // Split on the first comma only so review text may contain commas
            const commaIndex = lines[i].indexOf(',');
            if (commaIndex === -1) continue; // skip rows without a comma

            const brand = lines[i].slice(0, commaIndex).trim();
            const review = lines[i].slice(commaIndex + 1).trim();

            if (!brand || !review) continue; // skip empty cells

            if (!brandMap.has(brand)) {
              brandMap.set(brand, []);
            }
            brandMap.get(brand).push(review);
          }

          // Convert the Map to the standard array format
          parsedData = Array.from(brandMap.entries()).map(([brand, reviews]) => ({
            brand,
            reviews,
          }));

        } else if (fileName.endsWith('.txt')) {
          // ── TXT format ───────────────────────────────────────────────────
          // Each line: "BrandName: review text"
          const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);

          // Group lines by brand name into a Map
          const brandMap = new Map();
          for (const line of lines) {
            const colonIndex = line.indexOf(':');
            if (colonIndex === -1) continue; // skip lines without a colon

            const brand = line.slice(0, colonIndex).trim();
            const review = line.slice(colonIndex + 1).trim();

            if (!brand || !review) continue; // skip empty parts

            if (!brandMap.has(brand)) {
              brandMap.set(brand, []);
            }
            brandMap.get(brand).push(review);
          }

          // Convert the Map to the standard array format
          parsedData = Array.from(brandMap.entries()).map(([brand, reviews]) => ({
            brand,
            reviews,
          }));

        } else {
          // ── Unsupported format ───────────────────────────────────────────
          throw new Error(
            `Unsupported file format "${file.name}". Please upload a .json, .csv, or .txt file.`
          );
        }

        // ── Validate at least one brand with one review exists ────────────
        const hasContent = parsedData.some(
          (entry) => entry.reviews && entry.reviews.length > 0
        );

        if (!parsedData.length || !hasContent) {
          throw new Error(
            'The uploaded file contains no valid brand/review data. Please check the file content.'
          );
        }

        resolve(parsedData);

      } catch (err) {
        reject(err);
      }
    };

    reader.onerror = () => {
      reject(new Error('Failed to read the uploaded file. Please try again.'));
    };

    // Read the file as plain text (works for JSON, CSV, and TXT)
    reader.readAsText(file);
  });
}

// ─── Application Initializer ──────────────────────────────────────────────────

/**
 * Initializes the application:
 *   1. Loads review data from reviews.json (with error handling).
 *   2. Processes all reviews through the NLP pipeline.
 *   3. Activates the first brand to trigger the initial UI render.
 *   4. Wires the file upload button and input for dynamic data loading.
 *
 * Called automatically when the DOM is fully loaded.
 *
 * @returns {Promise<void>}
 *
 * Requirements: 1.1, 1.2, 1.3, 2.2, 13.3
 */
async function initApp() {
  // Step 1: Load review data from reviews.json; display error and abort on failure
  let rawData;
  try {
    rawData = await loadReviewData();
  } catch (err) {
    // Show the error banner and stop initialization — nothing else can render
    displayError(err.message);
    return;
  }

  // Step 2: Store the raw data in application state for later reference
  // (e.g., when merging uploaded data)
  appState.rawData = rawData;

  // Step 3: Run the full NLP pipeline over all brands and cache the results
  appState.processedReviews = processAllReviews(appState.rawData);

  // Step 4: Activate the first brand in the BRANDS list to trigger the initial
  // full UI render (brand selector, dashboard, filter bar, cards, word frequency)
  selectBrand(BRANDS[0]);

  // ── Wire the file upload button ──────────────────────────────────────────
  const uploadBtn  = document.getElementById('upload-btn');
  const fileInput  = document.getElementById('file-input');

  // Clicking the styled button programmatically opens the hidden file picker
  uploadBtn.addEventListener('click', () => fileInput.click());

  // Handle file selection
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return; // user cancelled the picker

    try {
      // Parse the uploaded file into the standard [{ brand, reviews }] format
      const uploadedData = await parseUploadedFile(file);

      // Merge uploaded brands into appState: add new brands, update existing ones
      for (const brandEntry of uploadedData) {
        // Add the raw entry so it's available for future re-processing
        appState.rawData.push(brandEntry);

        // Process only this brand's reviews and merge into the processed map
        const processed = processAllReviews([brandEntry]);
        appState.processedReviews.set(
          brandEntry.brand,
          processed.get(brandEntry.brand)
        );
      }

      // Re-render the brand selector with the updated full brand list
      // (includes both original BRANDS and any newly uploaded brands)
      const allBrands = [...appState.processedReviews.keys()];
      renderBrandSelector(allBrands, appState.activeBrand);

      // Activate the first uploaded brand so the user sees their data immediately
      selectBrand(uploadedData[0].brand);

      // Show a success toast confirming how many brands were loaded
      showToast(`✅ Loaded ${uploadedData.length} brand(s) from "${file.name}"`);

    } catch (err) {
      // Display the parse/validation error in the error banner
      displayError(err.message);
    }

    // Reset the file input so the same file can be re-uploaded if needed
    fileInput.value = '';
  });
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

// Start the application once the DOM is fully parsed and ready
document.addEventListener('DOMContentLoaded', initApp);
