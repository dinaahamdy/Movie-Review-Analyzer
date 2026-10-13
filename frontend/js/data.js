// Centralized movie dataset - 30 Real Movies with Official Posters
const movies = [
  // --- Science Fiction & Adventure ---
  {
    id: 'movie-1',
    title: 'Inception',
    year: 2010,
    genres: ['Science Fiction', 'Thriller'],
    poster: 'https://images.metahub.space/poster/medium/tt1375666/img',
    synopsis: 'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
    director: 'Christopher Nolan',
    duration: '148 min'
  },
  {
    id: 'movie-2',
    title: 'Interstellar',
    year: 2014,
    genres: ['Science Fiction', 'Adventure', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt0816692/img',
    synopsis: 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
    director: 'Christopher Nolan',
    duration: '169 min'
  },
  {
    id: 'movie-3',
    title: 'The Matrix',
    year: 1999,
    genres: ['Science Fiction', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0133093/img',
    synopsis: 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
    director: 'Lana Wachowski, Lilly Wachowski',
    duration: '136 min'
  },
  {
    id: 'movie-4',
    title: 'Dune',
    year: 2021,
    genres: ['Science Fiction', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt1160419/img',
    synopsis: 'A noble family becomes embroiled in a war for control over the galaxy\'s most valuable asset while its heir becomes troubled by visions of a dark future.',
    director: 'Denis Villeneuve',
    duration: '155 min'
  },
  {
    id: 'movie-5',
    title: 'Blade Runner 2049',
    year: 2017,
    genres: ['Science Fiction', 'Thriller'],
    poster: 'https://images.metahub.space/poster/medium/tt1856101/img',
    synopsis: 'Young Blade Runner K\'s discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard, who\'s been missing for thirty years.',
    director: 'Denis Villeneuve',
    duration: '164 min'
  },

  // --- Fantasy & Adventure ---
  {
    id: 'movie-6',
    title: 'Harry Potter and the Sorcerer\'s Stone',
    year: 2001,
    genres: ['Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0241527/img',
    synopsis: 'An orphaned boy enrolls in a school of wizardry, where he learns the truth about himself, his family and the terrible evil that haunts the magical world.',
    director: 'Chris Columbus',
    duration: '152 min'
  },
  {
    id: 'movie-7',
    title: 'The Lord of the Rings: The Fellowship of the Ring',
    year: 2001,
    genres: ['Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0120737/img',
    synopsis: 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron.',
    director: 'Peter Jackson',
    duration: '178 min'
  },
  {
    id: 'movie-8',
    title: 'Avatar',
    year: 2009,
    genres: ['Science Fiction', 'Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0499549/img',
    synopsis: 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
    director: 'James Cameron',
    duration: '162 min'
  },
  {
    id: 'movie-9',
    title: 'Pirates of the Caribbean: The Curse of the Black Pearl',
    year: 2003,
    genres: ['Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0325980/img',
    synopsis: 'Blacksmith Will Turner teams up with eccentric pirate "Captain" Jack Sparrow to save his love, the governor\'s daughter, from Jack\'s former pirate allies.',
    director: 'Gore Verbinski',
    duration: '143 min'
  },
  {
    id: 'movie-10',
    title: 'Jurassic Park',
    year: 1993,
    genres: ['Adventure', 'Science Fiction'],
    poster: 'https://images.metahub.space/poster/medium/tt0107290/img',
    synopsis: 'A pragmatic paleontologist touring an almost complete theme park on an island in Central America is tasked with protecting a couple of kids after a power failure causes the park\'s cloned dinosaurs to run loose.',
    director: 'Steven Spielberg',
    duration: '127 min'
  },

  // --- Drama & Romance ---
  {
    id: 'movie-11',
    title: 'Titanic',
    year: 1997,
    genres: ['Drama', 'Romance'],
    poster: 'https://images.metahub.space/poster/medium/tt0120338/img',
    synopsis: 'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
    director: 'James Cameron',
    duration: '194 min'
  },
  {
    id: 'movie-12',
    title: 'The Notebook',
    year: 2004,
    genres: ['Drama', 'Romance'],
    poster: 'https://images.metahub.space/poster/medium/tt0332280/img',
    synopsis: 'A poor yet passionate young man falls in love with a rich young woman, giving her a sense of freedom, but they are soon separated because of their social differences.',
    director: 'Nick Cassavetes',
    duration: '123 min'
  },
  {
    id: 'movie-13',
    title: 'Forrest Gump',
    year: 1994,
    genres: ['Drama', 'Romance'],
    poster: 'https://images.metahub.space/poster/medium/tt0109830/img',
    synopsis: 'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.',
    director: 'Robert Zemeckis',
    duration: '142 min'
  },
  {
    id: 'movie-14',
    title: 'The Shawshank Redemption',
    year: 1994,
    genres: ['Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt0111161/img',
    synopsis: 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
    director: 'Frank Darabont',
    duration: '142 min'
  },
  {
    id: 'movie-15',
    title: 'Pride & Prejudice',
    year: 2005,
    genres: ['Drama', 'Romance'],
    poster: 'https://images.metahub.space/poster/medium/tt0414387/img',
    synopsis: 'Sparks fly when spirited Elizabeth Bennet meets single, rich, and proud Mr. Darcy. But Mr. Darcy reluctantly finds himself falling in love with a woman beneath his class.',
    director: 'Joe Wright',
    duration: '129 min'
  },
  {
    id: 'movie-16',
    title: 'La La Land',
    year: 2016,
    genres: ['Drama', 'Romance'],
    poster: 'https://images.metahub.space/poster/medium/tt3783958/img',
    synopsis: 'While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.',
    director: 'Damien Chazelle',
    duration: '128 min'
  },

  // --- Thriller & Drama ---
  {
    id: 'movie-17',
    title: 'The Dark Knight',
    year: 2008,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt0468569/img',
    synopsis: 'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.',
    director: 'Christopher Nolan',
    duration: '152 min'
  },
  {
    id: 'movie-18',
    title: 'Joker',
    year: 2019,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt7286456/img',
    synopsis: 'In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime.',
    director: 'Todd Phillips',
    duration: '122 min'
  },
  {
    id: 'movie-19',
    title: 'Se7en',
    year: 1995,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt0114369/img',
    synopsis: 'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.',
    director: 'David Fincher',
    duration: '127 min'
  },
  {
    id: 'movie-20',
    title: 'Shutter Island',
    year: 2010,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt1130884/img',
    synopsis: 'In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane.',
    director: 'Martin Scorsese',
    duration: '138 min'
  },
  {
    id: 'movie-21',
    title: 'Parasite',
    year: 2019,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt6751668/img',
    synopsis: 'Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.',
    director: 'Bong Joon Ho',
    duration: '132 min'
  },

  // --- Animation & Family ---
  {
    id: 'movie-22',
    title: 'Spirited Away',
    year: 2001,
    genres: ['Animation', 'Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0245429/img',
    synopsis: 'During her family\'s move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, where humans are changed into beasts.',
    director: 'Hayao Miyazaki',
    duration: '125 min'
  },
  {
    id: 'movie-23',
    title: 'Spider-Man: Into the Spider-Verse',
    year: 2018,
    genres: ['Animation', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt4633694/img',
    synopsis: 'Teen Miles Morales becomes the Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions to stop a threat for all realities.',
    director: 'Bob Persichetti, Peter Ramsey',
    duration: '117 min'
  },
  {
    id: 'movie-24',
    title: 'Toy Story',
    year: 1995,
    genres: ['Animation', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0114709/img',
    synopsis: 'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy\'s room.',
    director: 'John Lasseter',
    duration: '81 min'
  },
  {
    id: 'movie-25',
    title: 'Up',
    year: 2009,
    genres: ['Animation', 'Adventure', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt1049413/img',
    synopsis: '78-year-old Carl Fredricksen travels to Paradise Falls in his house equipped with balloons, inadvertently taking a young stowaway.',
    director: 'Pete Docter, Bob Peterson',
    duration: '96 min'
  },
  {
    id: 'movie-26',
    title: 'Coco',
    year: 2017,
    genres: ['Animation', 'Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt2380307/img',
    synopsis: 'Aspiring musician Miguel, confronted with his family\'s ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer.',
    director: 'Lee Unkrich',
    duration: '105 min'
  },
  {
    id: 'movie-27',
    title: 'How to Train Your Dragon',
    year: 2010,
    genres: ['Animation', 'Fantasy', 'Adventure'],
    poster: 'https://images.metahub.space/poster/medium/tt0892769/img',
    synopsis: 'A hapless young Viking who aspires to hunt dragons becomes the unlikely friend of a young dragon himself, and learns there may be more to the creatures than he assumed.',
    director: 'Dean DeBlois, Chris Sanders',
    duration: '98 min'
  },

  // --- More Action & Thrillers ---
  {
    id: 'movie-28',
    title: 'Mad Max: Fury Road',
    year: 2015,
    genres: ['Science Fiction', 'Adventure', 'Thriller'],
    poster: 'https://images.metahub.space/poster/medium/tt1392190/img',
    synopsis: 'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners, a psychotic worshiper, and a drifter named Max.',
    director: 'George Miller',
    duration: '120 min'
  },
  {
    id: 'movie-29',
    title: 'The Silence of the Lambs',
    year: 1991,
    genres: ['Thriller', 'Drama'],
    poster: 'https://images.metahub.space/poster/medium/tt0102926/img',
    synopsis: 'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.',
    director: 'Jonathan Demme',
    duration: '118 min'
  },
  {
    id: 'movie-30',
    title: 'Avengers: Endgame',
    year: 2019,
    genres: ['Science Fiction', 'Adventure', 'Fantasy'],
    poster: 'https://images.metahub.space/poster/medium/tt4154796/img',
    synopsis: 'After the devastating events of Infinity War, the remaining Avengers assemble once more in order to reverse Thanos\' actions and restore balance to the universe.',
    director: 'Anthony Russo, Joe Russo',
    duration: '181 min'
  }
];