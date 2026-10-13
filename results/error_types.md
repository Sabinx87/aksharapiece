| Error type | Training corpus | AksharaPiece | Plain BPE |
|---|---|---|---|
| Same vowel sign twice | 25% (1386) | 18% (18) | 27% (18) |
| Nasal mark before vowel sign | 13% (731) | 2% (2) | 14% (9) |
| Sign after halant | 6% (317) | 21% (21) | 8% (5) |
| Two different vowel signs | 26% (1446) | 18% (18) | 18% (12) |
| Sign at start of word | 10% (582) | 24% (24) | 12% (8) |
| Other | 20% (1110) | 19% (19) | 21% (14) |
| **Total** | 5,572 | 102 | 66 |

Chi-square test, same mix of error types as the training corpus?

- AksharaPiece vs corpus: chi2 = 67.9, dof = 5, p = 2.79e-13
- Plain BPE vs corpus: chi2 = 2.3, dof = 5, p = 0.803
- Plain BPE vs AksharaPiece: chi2 = 17.3, dof = 5, p = 0.0039

**AksharaPiece examples**
- Same vowel sign twice: कुुशल (1), दुुई (1), यहाात (1), सुुधारो (1), कर्मैै (1), दुई-गृहोोड-नग्न (1)
- Nasal mark before vowel sign: आफंैले (1), गहँुदैन (1)
- Sign after halant: झ्ेख (1), सञ्ार (1), प्रस्ायन्त्रको (1), पश्ािषले (1), जनसंख््ध्याको (1), अध्ाइङ्ग (1)
- Two different vowel signs: विानुसारका (1), रहर्मन्नोेकाहरुको (1), पुजीे (1), पश्ािषले (1), विश््राियममा (1), मिडिेन (1)
- Sign at start of word: िु (5), ्न (3), ् (2), ीये (2), ेहरु (1), ितासहोदा (1)
- Other: (४५५ः४८ः१३।०१ः५८,९७७ः४१ः४७)। (5), "चतु्खाब" (1), पलाला्घुमा (1), लभ्ली्ठी, (1), कोला्रज (1), दि्छन्। (1)

**Plain BPE examples**
- Same vowel sign twice: गर्नुुपर्छ (2), सुुकुुुुुखले (2), उत्पत्तिित (1), सुुरु (1), बताउनुु (1), गर्नुुपर्ने (1)
- Nasal mark before vowel sign: बंैक (2), गुरूंै (1), एमालेँुले (1), काठमाडांैमा (1), हेर्‍यांै (1), चुुंगांै’ (1)
- Sign after halant: लाग््छ (1), अध्र्ारियों (1), &lsquo;butet्ाला (1), विस्ैति (1), सम्यक््यै (1)
- Two different vowel signs: मिस्रेकोे (1), समस्याित (1), नैवेद्यश्योेपित (1), वहाँलेै (1), सवारीिक (1), निद्राै (1)
- Sign at start of word: िन्होइटर (1), ुबाध्यलाग्दो (1), 'ु' (1), यो‌‌ंकृपाय (1), &lsquo;butet्ाला (1), प्रत्‍ाधीहरूको (1)
- Other: दि्छ। (4), निर्धा्रित (2), सिंगे्रसका (1), एण्टि्रयाक्ट (1), स्टेपबे्रक (1), ११ः४५ (1)
